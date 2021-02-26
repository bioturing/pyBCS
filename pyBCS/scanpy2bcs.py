import scanpy
import h5py
import numpy as np
import scipy
import os
import json
import pandas as pd
import uuid
import time
import shutil
import zipfile
from pandas.api.types import is_numeric_dtype
from abc import ABC, abstractmethod
import abc
BBROWSER_VERSION = "2.7.38"

class DataObject(ABC):
    def __init__(self, source, graph_based):
        """Constructor of DataObject class

        Keyword arguments:
            source: Path to input file or folder
            graph_based: Name of metadata that is the result of clustering process

        Returns:
            None
        """
        self.source = source
        self.graph_based = graph_based

    def get_n_cells(self):
        """Gets the number of cells

        Returns:
            The number of cells of the data
        """
        return len(self.get_barcodes())

    @abc.abstractclassmethod
    def get_barcodes(self):
        """Gets barcode names

        Returns:
            An array contains the barcode names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_barcodes(self):
        """Gets the raw barcode names

        Returns:
            An array contains the raw barcode names
        """
        pass

    @abc.abstractclassmethod
    def get_features(self):
        """Gets the gene names

        Returns:
            An array contains the gene names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_features(self):
        """Get the raw gene names

        Returns:
            An array contains the raw gene names
        """
        pass

    @abc.abstractclassmethod
    def get_raw_matrix(self):
        """Gets the raw matrix

        Returns:
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the matrix
        """
        pass

    @abc.abstractclassmethod
    def get_normalized_matrix(self):
        """Gets the normalized matrix

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the normalized matrix
        """
        pass

    def get_raw_data(self):
        """Gets raw data

        Returns:
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the matrix
            An array contains the raw barcode names
            An array contains the raw gene names
        """
        M = self.get_raw_matrix()
        barcodes = self.get_raw_barcodes()
        features = self.get_raw_features()
        return M, barcodes, features

    def get_normalized_data(self):
        """Gets the normalized data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the normalized matrix
            An array contains the barcode names
            An array contains the gene names
        """
        M = self.get_normalized_matrix()
        barcodes = self.get_barcodes()
        features = self.get_features()
        return M, barcodes, features

    @abc.abstractclassmethod
    def get_metadata(self):
        """Gets metadata

        Returns:
            A pandas.DataFrame contains the metadata
        """
        pass

    @abc.abstractclassmethod
    def get_dimred(self):
        """Gets dimentional reduction data

        Returns:
            A dictionary whose each value is a numpy.ndarray contains the dimentional reduced data
        """
        pass

    def sync_data(self, norm, raw):
        """Synces normalized and raw data

        Keyword arguments:
            norm: A tuple contains normalized data, which is the output of DataObject.get_normalized_data
            raw: A tuple contains raw data, which is the output of DataObject.get_raw_data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the synced normalized matrix
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the synced raw matrix
            An array contains the synced barcode names
            An array contains the synced gene names
            A boolean value indicates that if raw data is available
        """
        norm_M, norm_barcodes, norm_features = norm
        raw_M, raw_barcodes, raw_features = raw
        has_raw = True
        if raw_M is None:
            print("Raw data is not available, using normalized data as raw data", flush=True)
            raw_M = norm_M.tocsr()
            barcodes = norm_barcodes
            features = norm_features
            has_raw = False
        elif raw_M.shape == norm_M.shape:
            print("Normalized data and raw data are both available", flush=True)
            barcodes = norm_barcodes
            features = norm_features
        else:
            print("Shape of normalized data (%d, %d) does not equal shape of raw data (%d %d), using raw data as normalized data" % (norm_M.shape + raw_M.shape), flush=True)
            norm_M = raw_M.tocsc()
            barcodes = raw_barcodes
            features = raw_features
        return norm_M, raw_M, barcodes, features, has_raw

    def get_synced_data(self):
        """Gets synced version of normalized and raw data

        Returns:
            A scipy.sparse.csc_matrix of shape (cells x genes) contains the synced normalized matrix
            A scipy.sparse.csr_matrix of shape (cells x genes) contains the synced raw matrix
            An array contains the synced barcode names
            An array contains the synced gene names
            A boolean value indicates that if raw data is available
        """
        norm_M, norm_barcodes, norm_features = self.get_normalized_data()
        try:
            raw_M, raw_barcodes, raw_features = self.get_raw_data()
        except Exception as e:
            print("Cannot read raw data: %s" % str(e))
            raw_M = raw_barcodes = raw_features = None
        return self.sync_data((norm_M, norm_barcodes, norm_features),
                                    (raw_M, raw_barcodes, raw_features))

    def write_metadata(self, zobj, meta_path, replace_missing="Unassigned"):
        """Writes metadata to zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file
            replace_missing: A string indicates what missing values in metadata should be named

        Returns:
            None
        """
        print("Writing main/metadata/metalist.json")
        metadata = self.get_metadata()
        for metaname in metadata.columns:
            if metaname == self.graph_based:
                continue
            try:
                metadata[metaname] = pd.to_numeric(metadata[metaname],
                                                    downcast="float")
            except:
                print("Cannot convert %s to numeric, treating as categorical" % metaname)

        content = {}
        all_clusters = {}
        numeric_meta = metadata.select_dtypes(include=["number"]).columns
        category_meta = metadata.select_dtypes(include=["category"]).columns
        graph_based_uid = None
        for metaname in metadata.columns:
            uid = generate_uuid()

            if metaname in numeric_meta:
                all_clusters[uid] = list(metadata[metaname])
                lengths = 0
                names = "NaN"
                _type = "numeric"
            elif metaname in category_meta:
                if replace_missing not in metadata[metaname].cat.categories:
                    metadata[metaname] = add_category_to_first(metadata[metaname],
                                                                new_category=replace_missing)
                metadata[metaname].fillna(replace_missing, inplace=True)

                value_to_index = {}
                for x, y in enumerate(metadata[metaname].cat.categories):
                    value_to_index[y] = x
                all_clusters[uid] = [value_to_index[x] for x in metadata[metaname]]
                index, counts = np.unique(all_clusters[uid], return_counts = True)
                lengths = np.array([0] * len(metadata[metaname].cat.categories))
                lengths[index] = counts
                lengths = [x.item() for x in lengths]
                _type = "category"
                names = list(metadata[metaname].cat.categories)
            else:
                print("\"%s\" is not numeric or categorical, ignoring" % metaname)
                continue


            content[uid] = {
                "id":uid,
                "name":metaname,
                "clusterLength":lengths,
                "clusterName":names,
                "type":_type,
                "history":[generate_history_object()]
            }
            if metaname == self.graph_based:
                graph_based_uid = uid

        graph_based_history = generate_history_object()
        graph_based_history["hash_id"] = "graph_based"
        n_cells = self.get_n_cells()
        if graph_based_uid is not None:
            print("Found graph based clustering in metadata with keyword %s" % self.graph_based)
            content["graph_based"] = content[graph_based_uid].copy()
            content["graph_based"]["id"] = "graph_based"
            content["graph_based"]["name"] = "Graph based clusters"
            content["graph_based"]["history"] = [graph_based_history]
            all_clusters["graph_based"] = all_clusters[graph_based_uid]
            del content[graph_based_uid]
        else:
            print("Cannot find graph based clustering in metadata with keyword \"%s\", generating a fake one" % self.graph_based)
            content["graph_based"] = {
                "id":"graph_based",
                "name":"Graph-based clusters",
                "clusterLength":[0, n_cells],
                "clusterName":["Unassigned", "Cluster 1"],
                "type":"category",
                "history":[graph_based_history]
            }
            all_clusters["graph_based"] = [1] * n_cells
        with zobj.open(os.path.join(meta_path, "metalist.json"), "w") as z:
            z.write(json.dumps({"content":content, "version":1}).encode("utf8"))


        for uid in content:
            print("Writing main/metadata/%s.json" % uid, flush=True)
            obj = {
                "id":content[uid]["id"],
                "name":content[uid]["name"],
                "clusters":all_clusters[uid],
                "clusterName":content[uid]["clusterName"],
                "clusterLength":content[uid]["clusterLength"],
                "history":content[uid]["history"],
                "type":[content[uid]["type"]]
            }
            with zobj.open(os.path.join(meta_path, "%s.json" % uid), "w") as z:
                z.write(json.dumps(obj).encode("utf8"))

    def write_dimred_to_file(self, zobj, dimred_path, dimred_data):
        data = {}
        default_dimred = None
        if len(dimred_data.keys()) == 0:
            raise Exception("No dimred data found")
        for dimred in dimred_data:
            print("Writing %s" % dimred)
            matrix = dimred_data[dimred]
            if matrix.shape[1] > 3:
                print("%s has more than 3 dimensions, using only the first 3 of them" % dimred)
                matrix = matrix[:, 0:3]
            n_shapes = matrix.shape

            matrix = [list(map(float, x)) for x in matrix]
            dimred_history = generate_history_object()
            coords = {
                "coords":matrix,
                "name":dimred,
                "id":dimred_history["hash_id"],
                "size":list(n_shapes),
                "param":{"omics":"RNA", "dims":len(n_shapes)},
                "history":[dimred_history]
            }
            if default_dimred is None:
                default_dimred = coords["id"]
            data[coords["id"]] = {
                "name":coords["name"],
                "id":coords["id"],
                "size":coords["size"],
                "param":coords["param"],
                "history":coords["history"]
            }
            with zobj.open(os.path.join(dimred_path, coords["id"]), "w") as z:
                z.write(json.dumps(coords).encode("utf8"))
        meta = {
            "data":data,
            "version":1,
            "bbrowser_version":BBROWSER_VERSION,
            "default":default_dimred,
            "description":"Created by converting scanpy to bbrowser format"
        }
        print("Writing main/dimred/meta", flush=True)
        with zobj.open(os.path.join(dimred_path, "meta"), "w") as z:
            z.write(json.dumps(meta).encode("utf8"))

    def write_dimred(self, zobj, dimred_path):
        """Writes dimred data to the zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file

        Returns:
            None
        """
        print("Writing dimred")
        dimred_data = self.get_dimred()
        self.write_dimred_to_file(zobj, dimred_path, dimred_data)

    def write_matrix_to_hdf5(self, dest_hdf5, norm_M, raw_M, barcodes, features,
                                has_raw):
        print("Writing group \"bioturing\"")
        bioturing_group = dest_hdf5.create_group("bioturing")
        bioturing_group.create_dataset("barcodes",
                                        data=encode_strings(barcodes))
        bioturing_group.create_dataset("features",
                                        data=encode_strings(features))
        bioturing_group.create_dataset("data", data=raw_M.data)
        bioturing_group.create_dataset("indices", data=raw_M.indices)
        bioturing_group.create_dataset("indptr", data=raw_M.indptr)
        bioturing_group.create_dataset("feature_type", data=["RNA".encode("utf8")] * len(features))
        bioturing_group.create_dataset("shape", data=[len(features), len(barcodes)])

        if has_raw:
            print("Writing group \"countsT\"")
            raw_M_T = raw_M.tocsc()
            countsT_group = dest_hdf5.create_group("countsT")
            countsT_group.create_dataset("barcodes",
                                            data=encode_strings(features))
            countsT_group.create_dataset("features",
                                            data=encode_strings(barcodes))
            countsT_group.create_dataset("data", data=raw_M_T.data)
            countsT_group.create_dataset("indices", data=raw_M_T.indices)
            countsT_group.create_dataset("indptr", data=raw_M_T.indptr)
            countsT_group.create_dataset("shape", data=[len(barcodes), len(features)])
        else:
            print("Raw data is not available, ignoring \"countsT\"")

        print("Writing group \"normalizedT\"")
        normalizedT_group = dest_hdf5.create_group("normalizedT")
        normalizedT_group.create_dataset("barcodes",
                                        data=encode_strings(features))
        normalizedT_group.create_dataset("features",
                                        data=encode_strings(barcodes))
        normalizedT_group.create_dataset("data", data=norm_M.data)
        normalizedT_group.create_dataset("indices", data=norm_M.indices)
        normalizedT_group.create_dataset("indptr", data=norm_M.indptr)
        normalizedT_group.create_dataset("shape", data=[len(barcodes), len(features)])

        print("Writing group \"colsum\"")
        norm_M = norm_M.tocsr()
        n_cells = len(barcodes)
        sum_lognorm = np.array([0.0] * n_cells)
        if has_raw:
            sum_log = np.array([0.0] * n_cells)
            sum_raw = np.array([0.0] * n_cells)

        for i in range(n_cells):
            l, r = raw_M.indptr[i:i+2]
            sum_lognorm[i] = np.sum(norm_M.data[l:r])
            if has_raw:
                sum_raw[i] = np.sum(raw_M.data[l:r])
                sum_log[i] = np.sum(np.log2(raw_M.data[l:r] + 1))

        colsum_group = dest_hdf5.create_group("colsum")
        colsum_group.create_dataset("lognorm", data=sum_lognorm)
        if has_raw:
            colsum_group.create_dataset("log", data=sum_log)
            colsum_group.create_dataset("raw", data=sum_raw)

    def write_matrix(self, dest_hdf5):
        """Writes expression data to the zip file

        Keyword arguments:
            zobj: The opened-for-write zip file
            dest_hdf5: An opened-for-write hdf5 file where the expression should be written to

        Returns:
            An array that contains the barcode names that are actually written to the hdf5 file
            An array that contains the gene names that are actually written to the hdf5 file
            A boolean indicates that if raw data is available
        """
        #TODO: Reduce memory usage
        norm_M, raw_M, barcodes, features, has_raw = self.get_synced_data()
        self.write_matrix_to_hdf5(dest_hdf5, norm_M, raw_M, barcodes, features,
                                    has_raw)
        return barcodes, features, has_raw



    def write_main_folder_to_file(self, zobj, main_path, matrix, barcodes, features):
        print("Writing to zip", flush=True)
        zobj.write(matrix, os.path.join(main_path, "matrix.hdf5"))

        print("Writing main/barcodes.tsv", flush=True)
        with zobj.open(os.path.join(main_path, "barcodes.tsv"), "w") as z:
            z.write("\n".join(barcodes).encode("utf8"))

        print("Writing main/genes.tsv", flush=True)
        with zobj.open(os.path.join(main_path, "genes.tsv"), "w") as z:
            z.write("\n".join(features).encode("utf8"))

        print("Writing main/gene_gallery.json", flush=True)
        with zobj.open(os.path.join(main_path, "gene_gallery.json"), "w") as z:
            z.write(json.dumps(self.get_gene_gallery_object()).encode("utf8"))

    def write_main_folder(self, zobj, main_path):
        """Writes data to "main" folder

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file

        Returns:
            A boolean indicates that if raw data is available
        """
        print("Writing main/matrix.hdf5", flush=True)
        tmp_matrix = "." + str(uuid.uuid4())
        with h5py.File(tmp_matrix, "w") as dest_hdf5:
            barcodes, features, has_raw = self.write_matrix(dest_hdf5)
        self.write_main_folder_to_file(zobj, main_path, tmp_matrix, barcodes,
                                        features)
        os.remove(tmp_matrix)
        return has_raw

    def write_runinfo(self, zobj, study_name, runinfo_path, unit):
        """Writes run_info.json

        Keyword arguments:
            zobj: The opened-for-write zip file
            root_name: Name of the root directory that contains the whole study in the zip file
            unit: Unit of the study

        Returns:
            None
        """
        print("Writing run_info.json", flush=True)
        runinfo_history = generate_history_object()
        runinfo_history["hash_id"] = study_name
        date = time.time() * 1000
        run_info = {
            "species":"human",
            "hash_id":study_name,
            "version":16,
            "n_cell":self.get_n_cells(),
            "modified_date":date,
            "created_date":date,
            "addon":"SingleCell",
            "matrix_type":"single",
            "n_batch":1,
            "platform":"unknown",
            "omics":["RNA"],
            "title":["Created by bbrowser converter"],
            "history":[runinfo_history],
            "unit":unit
        }
        with zobj.open(os.path.join(runinfo_path, "run_info.json"), "w") as z:
            z.write(json.dumps(run_info).encode("utf8"))

    def write_bcs_to_file(self, zobj, study_name, replace_missing):
        self.write_metadata(zobj, meta_path=self.get_metadata_path(study_name),
                            replace_missing=replace_missing)
        self.write_dimred(zobj, dimred_path=self.get_dimred_path(study_name))
        has_raw = self.write_main_folder(zobj,
                                    main_path=self.get_main_path(study_name))
        unit = "umi" if has_raw else "lognorm"
        self.write_runinfo(zobj, study_name=study_name,
                            runinfo_path=self.get_runinfo_path(study_name),
                            unit=unit)

    def write_bcs(self, study_name, output_name, replace_missing="Unassigned"):
        """Writes data to bcs file

        Keyword arguments:
            root_name: Name of the root directory that contains the whole study in the zip file
            output_name: Path to output file
            replace_missing: A string indicates what missing values in metadata should be named

        Returns:
            Path to output file
        """
        with zipfile.ZipFile(output_name, "w") as zobj:
            self.write_bcs_to_file(zobj, study_name, replace_missing)
        return output_name

    def get_metadata_path(self, study_name):
        return os.path.join(study_name, "main", "metadata")

    def get_gene_gallery_object(self):
        obj = {"gene":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"gene"},"version":1,"protein":{"nameArr":[],"geneIDArr":[],"hashID":[],"featureType":"protein"}}
        return obj

    def get_dimred_path(self, study_name):
        return os.path.join(study_name, "main", "dimred")

    def get_main_path(self, study_name):
        return os.path.join(study_name, "main")

    def get_runinfo_path(self, study_name):
        return study_name

class ScanpyData(DataObject):
    def __init__(self, source, graph_based, raw_key="counts"):
        """Constructor of ScanpyData object

        Keyword arguments:
            source: Path to input file or folder
            graph_based: Name of metadata that is the result of clustering process
            raw_key: Where to look for raw data in AnnData.layers
        """
        DataObject.__init__(self, source=source,
                            graph_based="louvain" if graph_based is None else graph_based)
        self.object = scanpy.read_h5ad(source, "r")
        self.raw_key = raw_key

    def get_barcodes(self):
        return self.object.obs_names

    def get_features(self):
        return self.object.var_names

    def get_raw_barcodes(self):
        return self.get_barcodes()

    def get_raw_features(self):
        try:
            return self.object.raw.var.index
        except:
            return self.get_features()

    def get_raw_matrix(self):
        try:
            return self.object.raw.X[:][:].tocsr()
        except:
            return self.object.layers[self.raw_key].tocsr()

    def get_normalized_matrix(self):
        return self.object.X[:][:].tocsc()

    def get_metadata(self):
        return self.object.obs

    def get_dimred(self):
        res = {}
        for dimred in self.object.obsm:
            if isinstance(self.object.obsm[dimred], np.ndarray) == False:
                print("%s is not a numpy.ndarray, ignoring" % dimred)
                continue
            res[dimred] = self.object.obsm[dimred]
        return res

class SubclusterData(DataObject):
    @abc.abstractclassmethod
    def get_sub_barcodes(self, sub_name):
        pass

    @abc.abstractclassmethod
    def get_sub_raw_barcodes(self, sub_name):
        pass

    @abc.abstractclassmethod
    def get_sub_raw_matrix(self, sub_name):
        pass

    @abc.abstractclassmethod
    def get_sub_normalized_matrix(self, sub_name):
        pass

    def get_sub_normalized_data(self, sub_name):
        M = self.get_sub_normalized_matrix(sub_name)
        barcodes = self.get_sub_barcodes(sub_name)
        features = self.get_features()
        return M, barcodes, features

    def get_sub_raw_data(self, sub_name):
        M = self.get_sub_raw_matrix(sub_name)
        barcodes = self.get_sub_raw_barcodes(sub_name)
        features = self.get_features()
        return M, barcodes, features

    def write_sub_folder(self, zobj, sub_path, sub_name):
        print("Writing sub/%s/matrix.hdf5" % sub_name)
        tmp_matrix = "." + str(uuid.uuid4())
        with h5py.File(tmp_matrix, "w") as dest_hdf5:
            barcodes, features, has_raw = self.write_sub_matrix(sub_name,
                                                                dest_hdf5)
        self.write_main_folder_to_file(zobj, sub_path, tmp_matrix, barcodes,
                                        features)
        os.remove(tmp_matrix)
        return has_raw

    def write_sub_matrix(self, sub_name, dest_hdf5):
        norm_M, raw_M, barcodes, features, has_raw = self.get_sub_synced_data(sub_name)
        self.write_matrix_to_hdf5(dest_hdf5, norm_M, raw_M, barcodes, features,
                                    has_raw)
        return barcodes, features, has_raw

    def get_sub_synced_data(self, sub_name):
        #TODO should not read from file everytime
        norm_M, raw_M, barcodes, features, has_raw = self.get_synced_data()
        ids = self.get_sub_cell_indexes(sub_name)
        return norm_M[ids, :], raw_M[ids, :], barcodes[ids], features, has_raw

    @abc.abstractclassmethod
    def get_sub_dimred(self, sub_name):
        pass

    def write_sub_dimred(self, zobj, sub_dimred_path, sub_name):
        print("Writing dimred of subcluster %s" % sub_name)
        dimred = self.get_sub_dimred(sub_name)
        self.write_dimred_to_file(zobj, dimred_path=sub_dimred_path,
                                    dimred_data=dimred)

    @abc.abstractclassmethod
    def get_sub_cluster_names(self):
        pass

    @abc.abstractclassmethod
    def get_sub_cell_indexes(self, sub_name):
        pass

    def write_cluster_info(self, zobj, cluster_info_path, sub_name, selected_arr):
        obj = {
                "name":sub_name,
                "hash":sub_name,
                "selectedArr":selected_arr,
                #TODO Use a defined constant for this
                "bbrowser_version":BBROWSER_VERSION,
                "created_at":time.time() * 1000,
                "created_by":"bbrowser_format_converter",
                "version":1,
                "path":sub_name
                }
        with zobj.open(os.path.join(cluster_info_path, "cluster_info.json"), "w") as z:
            z.write(json.dumps(obj).encode("utf8"))

    def write_sub_clusters(self, zobj, study_name):
        sub_cluster_names = self.get_sub_cluster_names()
        id_list = []
        for cluster in sub_cluster_names:
            cell_indexes = self.get_sub_cell_indexes(cluster)
            #TODO Change this to hash in the future
            cluster_id = cluster
            id_list.append(cluster_id)
            self.write_sub_dimred(zobj, self.get_sub_dimred_path(study_name, cluster_id),
                                    sub_name=cluster_id)
            self.write_sub_folder(zobj, self.get_sub_path(study_name, cluster_id),
                                    sub_name=cluster_id)
            tmp = [int(x) for x in cell_indexes]
            self.write_cluster_info(zobj, self.get_cluster_info_path(study_name, cluster_id),
                                    sub_name=cluster_id, selected_arr=tmp)
        with zobj.open(os.path.join(study_name, "sub", "graph_cluster.json"), "w") as z:
            z.write(json.dumps({"main":id_list}).encode("utf8"))

    def write_bcs_to_file(self, zobj, study_name, replace_missing):
        DataObject.write_bcs_to_file(self, zobj, study_name, replace_missing)
        self.write_sub_clusters(zobj, study_name=study_name)

    def get_sub_dimred_path(self, study_name, sub_name):
        return os.path.join(study_name, "sub", sub_name, "dimred")

    def get_sub_path(self, study_name, sub_name):
        return os.path.join(study_name, "sub", sub_name)

    def get_cluster_info_path(self, study_name, sub_name):
        return os.path.join(study_name, "sub", sub_name)

class SpringData(SubclusterData):
    def __init__(self, source, graph_based):
        """Constructor of SpringData object

        Keyword arguments:
            source: Path to input file or folder
            graph_based: Name of metadata that is the result of clustering process
        """
        DataObject.__init__(self, source=source,
                            graph_based="ClustersWT" if graph_based is None else graph_based)

    def get_barcodes(self):
        full_data = self.get_full_data_names()[0]
        idx = self.get_sub_cell_indexes(full_data)
        return np.array(idx).astype("str")

    def get_raw_barcodes(self):
        return None

    def get_features(self):
        with open(os.path.join(self.source, "genes.txt"), "r") as f:
            lines = f.readlines()
        return [x[0:-1].rsplit("_", 1)[0] for x in lines]

    def get_raw_features(self):
        return None

    def get_raw_matrix(self):
        return None

    def get_normalized_matrix(self):
        with np.load(os.path.join(self.source, "counts_norm.npz"), "r") as f:
            sparse_format = f["format"].astype("str")
            if sparse_format == "csc":
                return scipy.sparse.csc_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])
            elif sparse_format == "csr":
                return scipy.sparse.csr_matrix((f["data"], f["indices"], f["indptr"]),
                                                shape=f["shape"])\
                                    .tocsc()
            else:
                raise Exception("Format %s is not supported" % sparse_format)

    def get_metadata(self):
        full_data = self.get_full_data_names()[0]
        with open(os.path.join(self.source, full_data,
                                "categorical_coloring_data.json"),
                    "r") as f:
            obj = json.load(f)
        for key in obj:
            obj[key] = obj[key]["label_list"]
        metadata = pd.DataFrame.from_dict(obj)
        for meta in metadata:
            if len(np.unique(metadata[meta])) < len(metadata[meta]):
                metadata[meta] = pd.Series(metadata[meta], dtype="category")
        return metadata

    def get_dimred(self):
        full_data_names = self.get_full_data_names()
        dimred = {}
        for data_name in full_data_names:
            d = self.get_sub_dimred(data_name)
            #TODO: Avoid hard coding "coordinates"?
            dimred[data_name] = d["coordinates"]
        return dimred

    def get_sub_barcodes(self, sub_name):
        ids = self.get_sub_cell_indexes(sub_name)
        return np.array(self.get_barcodes())[ids]

    def get_sub_raw_barcodes(self, sub_name):
        ids = self.get_sub_cell_indexes(sub_name)
        return np.array(self.get_raw_barcodes())[ids]

    def get_sub_normalized_matrix(self, sub_name):
        ids = self.get_sub_cell_indexes(sub_name)
        return self.get_normalized_matrix()[ids, :]

    def get_sub_raw_matrix(self, sub_name):
        ids = self.get_sub_cell_indexes(sub_name)
        return self.get_raw_matrix()[ids, :]

    def get_sub_data_names(self):
        dirs = os.listdir(self.source)
        res = []
        for d in dirs:
            if os.path.isdir(os.path.join(self.source, d)):
                x = os.path.join(self.source, d, "run_info.json")
                if os.path.exists(x):
                    res.append(d)
        return res

    def get_sub_cluster_names(self):
        sub_data_names = self.get_sub_data_names()
        res = []
        for d in sub_data_names:
            if self.is_full_data(d) == False:
                res.append(d)
        return res

    def get_full_data_names(self):
        sub_data_names = self.get_sub_data_names()
        res = []
        for d in sub_data_names:
            if self.is_full_data(d):
                res.append(d)
        return res

    def is_full_data(self, data_name):
        return data_name.startswith("FullDataset_v1")

    def get_sub_cell_indexes(self, sub_name):
        try:
            idx = np.load(os.path.join(self.source, sub_name, "cell_filter.npy"))
        except:
            with open(os.path.join(self.source, sub_name, "cell_filter.txt"),
                        "r") as f:
                lines = f.readlines()
            idx = np.array([int(x[0:-1]) for x in lines])
        return idx

    def get_sub_dimred(self, sub_name):
        df = pd.read_csv(os.path.join(self.source, sub_name, "coordinates.txt"),
                            names=["index", "x", "y"],
                            index_col="index")
        df.index = df.index.map(str)
        coordinates = df.loc[self.get_barcodes(), :].to_numpy()
        return {"coordinates" : coordinates}

def generate_uuid(remove_hyphen=True):
    """Generates a unique uuid string

    Keyword arguments:
        remove_hyphen: True if the hyphens should be removed from the uuid, False otherwise
    """
    res = str(uuid.uuid4())
    if remove_hyphen == True:
        res = res.replace("-", "")
    return res

def encode_strings(strings, encode_format="utf8"):
    """Converts an array/list of strings into utf8 representation
    """
    return [x.encode(encode_format) for x in strings]

def generate_history_object():
    """Generates a Bioturing-format history object
    """
    return {
        "created_by":"bbrowser_format_converter",
        "created_at":time.time() * 1000,
        "hash_id":generate_uuid(),
        "description":"Created by converting scanpy object to bbrowser format"
    }

def add_category_to_first(column, new_category):
    """Adds a new category to a pd.Categorical object

    Keyword arguments:
        column: The pd.Categorical object
        new_category: The new category to be added

    Returns:
        A new pd.Categorical object that is almost the same as the given one,
            except for a new category is added (if it is not already included in the original object).
            The new category is added to first in the categories list.
    """
    if column.dtype.name != "category":
        raise Exception("Object is not a pandas.Categorical")

    if new_category in column.cat.categories:
        raise Exception("%s is already in categories list" % new_category)

    column = column.copy()
    column = column.cat.add_categories(new_category)
    cat = column.cat.categories.tolist()
    cat = cat[0:-1]
    cat.insert(0, new_category)
    column = column.cat.reorder_categories(cat)
    return column

def format_data(source, output_name, input_format="h5ad", raw_key="counts", replace_missing="Unassigned", graph_based=None):
    """Converts data to bcs format

    Keyword arguments:
        source: Path to input file or folder
        output_name: Path to output file
        input_format: Format of input file, must be either "h5ad" or "spring"
        raw_keys: If input_format is "h5ad", raw_keys specifies where to look for raw data in AnnData.layers.
                    If raw data is stored in AnnData.raw instead, this parameter is ignored.
        replace_missing: A string indicates what missing values in metadata should be named
        graph_based: A string that indicates the name of the metadata that is the result of clustering process.
                        If None is given, the program looks for "louvain" when input format is "h5ad"
                            or "ClustersWT" when input format is "spring".
                        If graph_based is not in the metadata, a dummy graph based is generated where all cells
                            belong to "Cluster 1".

    Returns:
        Path to output file
    """
    study_id = generate_uuid(remove_hyphen=False)
    if input_format == "h5ad":
        data_object = ScanpyData(source, raw_key=raw_key, graph_based=graph_based)
    elif input_format == "spring":
        data_object = SpringData(source, graph_based=graph_based)
    else:
        raise Exception("Invalid input format: %s" % input_format)
    return data_object.write_bcs(study_name=study_id, output_name=output_name,
                                    replace_missing=replace_missing)

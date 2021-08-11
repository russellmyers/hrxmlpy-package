import pandas as pd
import numpy as np
import time


class SKDecisionForestAnalyser:
    """
    Used to analyse standard scikit learn decision tree classifiers  to determine features most used for each sample.

    Main entry point: determine_important_features()

    """

    def __init__(self, trained_model, feat_name_list=[]):
        """
        :param trained_model: Needs to be a model based on scikit learn DecsionTreeClassifer
        :param feat_name_list: Feature names
        """
        self.trained_model = trained_model
        self.feat_name_list = feat_name_list

    def _get_indices(self, est, samples):
        """ Tree based learning algorithms only """

        indices_ar = []
        dec_paths = est.decision_path(samples)
        for i in range(0, len(samples)):
            indices_ar.append(dec_paths.indices[dec_paths.indptr[i] : dec_paths.indptr[i + 1]])
        return np.array(indices_ar)

    def _is_leaf(self, tr, ind):
        """ Tree based learning algorithms only """

        if (tr.children_left[ind] == -1) and (tr.children_right[ind] == -1):
            return True

        return False

    def _get_parent_ind_all_inds(self, tr):
        """ Tree based learning algorithms only """

        ch_left = tr.children_left
        ch_right = tr.children_right

        parent_dict = {}

        for p_num, child_num in enumerate(ch_left):
            if child_num == -1:
                pass
            else:
                parent_dict[child_num] = p_num

        for p_num, child_num in enumerate(ch_right):
            if child_num == -1:
                pass
            else:
                parent_dict[child_num] = p_num

        return parent_dict

    def _get_parent_ind(self, tr, ind, parent_dict=None):
        """ Tree based learning algorithms only """

        if parent_dict is None:
            ch_left = tr.children_left
            ch_right = tr.children_right
            for i, entry in enumerate(ch_left):
                if entry == ind:
                    return i
            for i, entry in enumerate(ch_right):
                if entry == ind:
                    return i
        else:
            return parent_dict[ind]

    def _get_feature_used(self, tr, ind, parent_dict=None):
        """ Tree based learning algorithms only """

        f_used = tr.feature[ind]
        if f_used == -2:
            p_id = self._get_parent_ind(tr, ind, parent_dict)
            return tr.feature[p_id]
        else:
            return f_used

    def _process_est_tree_sample(
        self, sample_num, indices, tr, feature_used_list_all_trees, last_feature_used_list_all_trees, parent_dict
    ):
        """ Tree based learning algorithms only """

        operations_this_sample = 0
        feature_used_dict = feature_used_list_all_trees[sample_num]
        last_feature_used_dict = last_feature_used_list_all_trees[sample_num]
        for ind in indices:
            operations_this_sample += 1
            if self._is_leaf(tr, ind):
                last_feat = self._get_feature_used(tr, ind, parent_dict)
                if last_feat in last_feature_used_dict:
                    last_feature_used_dict[last_feat] += 1
                else:
                    last_feature_used_dict[last_feat] = 1

            f_used = self._get_feature_used(tr, ind, parent_dict)
            if f_used in feature_used_dict:
                feature_used_dict[f_used] += 1
            else:
                feature_used_dict[f_used] = 1

        return operations_this_sample

    def _process_est_tree(
        self,
        tree_num,
        est,
        samples,
        feature_used_list_all_trees,
        last_feature_used_list_all_trees,
        show_progress_after_each=-1,
        show_diagnostics_flag=False,
    ):
        """ Tree based learning algorithms only """

        startTree = int(time.time() * 1000)
        if show_progress_after_each != -1:
            if tree_num % show_progress_after_each == 0:
                print("Processing tree num: ", tree_num + 1, " / ", len(self.trained_model.estimators_))
        tr = est.tree_
        ind_sets = self._get_indices(est, samples)
        parent_dict = self._get_parent_ind_all_inds(tr)
        tot_operations = 0

        for sample_num, indices in enumerate(ind_sets):
            tot_operations += self._process_est_tree_sample(
                sample_num, indices, tr, feature_used_list_all_trees, last_feature_used_list_all_trees, parent_dict
            )

        end_tree = int(time.time() * 1000)

        if (show_progress_after_each != -1) and (show_diagnostics_flag):
            if tree_num % show_progress_after_each == 0:
                print(
                    "Tree: %s took: %s ms. Tot ops performed: %s" % (tree_num + 1, end_tree - startTree, tot_operations)
                )

    def _accum_features_used(
        self, df_X_scaled, show_progress_after_each=-1, show_diagnostics_flag=False, num_trees_to_check=None
    ):
        """ Tree based learning algorithms only """
        # set num_trees_to_check to a specific number if wanting to restrict to a few trees for testing purposes

        start_accum = int(time.time() * 1000)
        i = 0

        samples = df_X_scaled.values
        print("\nDetermining important features per sample")
        feature_used_list_all_trees = [{} for i in range(0, len(samples))]
        last_feature_used_list_all_trees = [{} for i in range(0, len(samples))]
        for tree_num, est in enumerate(self.trained_model.estimators_):
            if num_trees_to_check is None:
                pass
            else:
                if tree_num > num_trees_to_check:
                    break

            self._process_est_tree(
                tree_num,
                est,
                samples,
                feature_used_list_all_trees,
                last_feature_used_list_all_trees,
                show_progress_after_each=show_progress_after_each,
                show_diagnostics_flag=show_diagnostics_flag,
            )

        end_accum = int(time.time() * 1000)
        if show_diagnostics_flag:
            print("Overall time in accum: ", end_accum - start_accum)

        return feature_used_list_all_trees, last_feature_used_list_all_trees

    def _top_features_used(self, fu_array, feat_name_list, num_top_features=3):
        """ Tree based learning algorithms only """
        sorted_ar = []

        feat_name_list_ext = feat_name_list[:]
        feat_name_list_ext.append(".")  # Use when no top feature is available for a sample

        min_top_feats = 9999

        for entry in fu_array:
            tmp = sorted(entry, key=entry.get, reverse=True)
            if len(tmp) < min_top_feats:
                min_top_feats = len(tmp)
            if len(tmp) < num_top_features:
                tmp.extend([len(feat_name_list_ext) - 1 for n in range(len(tmp), num_top_features)])
            sorted_ar.append(tmp[:num_top_features])

        print("Min top features: ", min_top_feats)
        named_ar = []
        for entry in sorted_ar:
            new_entry = [feat_name_list_ext[x] for x in entry]
            named_ar.append(new_entry)
        return np.array(named_ar)

    def determine_important_features(
        self, df_X_scaled, df, num_top_features, show_progress_after_each=20, show_diagnostics_flag=False
    ):
        """

        :param df_X_scaled:  scaled df with features only
        :param df:  master df (can include other id columns)
        :param num_top_features: How many most important features to return per sample
        :param show_progress_after_each:
        :param show_diagnostics_flag:
        :return: df with most important feature columns added
        """

        feat_name_list = self.feat_name_list

        fu = self._accum_features_used(
            df_X_scaled, show_progress_after_each, show_diagnostics_flag, num_trees_to_check=None
        )

        if show_diagnostics_flag:
            print(" ")
            print("First 5 samples total feature usage: ")
            for entry in fu[0][:5]:
                print(entry)
            print(" ")
            print("First 5 samples last feature usage: ")
            for entry in fu[1][:5]:
                print(entry)

        # Check top features used per example over all tree branches
        fu_overall = self._top_features_used(fu[0], feat_name_list, num_top_features)
        if show_diagnostics_flag:
            print(" ")
            print("First 5 samples top n overall features used: ")
            for entry in fu_overall[:5]:
                print(entry)

        # Check top features used  per example as last branch only of tree path
        fu_last = self._top_features_used(fu[1], feat_name_list, num_top_features)
        if show_diagnostics_flag:
            print(" ")
            print("First 5 samples top n last features used: ")
            for entry in fu_last[:5]:
                print(entry)

        df_with_feature_importances = df.copy()

        for i in range(0, num_top_features):
            df_with_feature_importances["F" + str(i + 1)] = fu_overall[:, i]

        for i in range(0, num_top_features):
            df_with_feature_importances["LF" + str(i + 1)] = fu_last[:, i]

        return df_with_feature_importances

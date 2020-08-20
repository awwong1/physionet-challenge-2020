import json
from neurokit2_parallel import KEYS_INTERVALRELATED


def parse_fc_parameters(important_features):
    """Takes importances_rank.json, 'sorted_keys' value,
    returns tsfresh compatible fc_parameters dictionary by lead
    """

    per_lead_fc_params = {}
    for important_feature in sorted(important_features):
        lead_w_feat_type, *tsfresh_config = important_feature.split("__")

        # lead, feat_type = lead_w_feat_type.split("_")
        lead_feat_type_fc = per_lead_fc_params.get(lead_w_feat_type, {})
        if len(tsfresh_config) == 0:
            # this is an interval related HRV metric, or meta
            assert any(
                key in lead_w_feat_type for key in KEYS_INTERVALRELATED + ["age", "sex"]
            ), f"Invalid feature: {important_feature}"
        elif len(tsfresh_config) == 1:
            lead_feat_type_fc[tsfresh_config[0]] = None
        else:
            feat_key = tsfresh_config[0]
            feat_args = tsfresh_config[1:]

            lead_feat_list = lead_feat_type_fc.get(feat_key, [])
            lead_feat_dict = {}

            for feat_arg in feat_args:
                *feat_arg_key, feat_arg_val = feat_arg.split("_")
                feat_arg_key = "_".join(feat_arg_key)

                try:
                    if feat_arg_val.startswith("(") and feat_arg_val.endswith(")"):
                        # check for tuple
                        feat_arg_val = tuple(map(int, feat_arg_val.replace('(','').replace(')','').split(',')))
                    else:
                        # parse string, float, int
                        feat_arg_val = json.loads(feat_arg_val)
                except ValueError:
                    # check for boolean
                    if feat_arg_val == "True":
                        feat_arg_val = True
                    elif feat_arg_val == "False":
                        feat_arg_val = False

                lead_feat_dict[feat_arg_key] = feat_arg_val
            lead_feat_list.append(lead_feat_dict)
            lead_feat_type_fc[feat_key] = lead_feat_list

        if len(tsfresh_config) > 0:
            per_lead_fc_params[lead_w_feat_type] = lead_feat_type_fc
        else:
            per_lead_fc_params[lead_w_feat_type] = None

    return per_lead_fc_params

import pandas as pd
import numpy as np
from os import listdir
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def get_dataframe():
    files = [f"./data/{file}" for file in listdir("./data")]
    export_filename = ""
    survey_one_name = ""
    survery_two_name = ""
    for file in files:
        if file.startswith("./data/DBS_Database_Parkinson_export_"):
            export_filename = file
        elif file.startswith(
            "./data/DBS_Database_Parkinson_Vragenlijst_pre-operatief_export_"
        ):
            survey_one_name = file
        elif file.startswith(
            "./data/DBS_Database_Parkinson_Vragenlijst_post-operatief_export_"
        ):
            survery_two_name = file
    original_data = pd.read_csv(export_filename, sep=";", low_memory=False)
    survery_one = pd.read_csv(survey_one_name, sep=";", low_memory=False)
    survery_two = pd.read_csv(survery_two_name, sep=";", low_memory=False)
    original_data = original_data.merge(
        survery_one,
        left_on="Participant Id",
        right_on="Castor Participant ID",
        how="outer",
    )
    original_data = original_data.merge(
        survery_two,
        left_on="Participant Id",
        right_on="Castor Participant ID",
        how="outer",
    )

    original_data = original_data[original_data["Site Abbreviation"] != "TES"]
    original_data.replace(
        [-99, -98, -97, -96, -95, "-99", "-98", "-97", "-96", "-95", "01-01-2999"],
        np.nan,
        inplace=True,
    )
    original_data["PRE_TOT_LED"].replace(0, np.nan, inplace=True)
    original_data["POST_TOT_LED"].replace(0, np.nan, inplace=True)
    original_data = original_data.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # scrambles the records

    new_data = pd.DataFrame()

    new_data[["SCREEN_DT", "SURGERY_DT", "OPTIM_STIMULPAR_DT"]] = original_data[
        ["SCREEN_DT", "SURGERY_DT", "OPTIM_STIMULPAR_DT"]
    ].apply(pd.to_datetime, format="%d-%m-%Y")

    new_data[["BIRTH_YR", "PARKINSON_YR"]] = original_data[
        ["BIRTH_YR", "PARKINSON_YR"]
    ].apply(pd.to_datetime, format="%Y")

    new_data["Days between screening and surgery"] = (
        new_data["SURGERY_DT"] - new_data["SCREEN_DT"]
    ).dt.days
    new_data["Days between surgery and follow up"] = (
        new_data["OPTIM_STIMULPAR_DT"] - new_data["SURGERY_DT"]
    ).dt.days
    new_data["Days between screening and follow up"] = (
        new_data["OPTIM_STIMULPAR_DT"] - new_data["SCREEN_DT"]
    ).dt.days

    new_data["AGE"] = (new_data["SCREEN_DT"] - new_data["BIRTH_YR"]).dt.days // 365
    new_data["ziekteduur_calc"] = (
        new_data["SCREEN_DT"] - new_data["PARKINSON_YR"]
    ).dt.days // 365

    new_data["TOT_pre_UPDRS_I"] = (
        original_data["pre_UPDRS_1_1"]
        + original_data["pre_UPDRS_1_2"]
        + original_data["pre_UPDRS_1_3"]
        + original_data["pre_UPDRS_1_4"]
        + original_data["pre_UPDRS_1_5"]
        + original_data["pre_UPDRS_1_6"]
        + original_data["pre_UPDRS_1_7"]
        + original_data["pre_UPDRS_1_8"]
        + original_data["pre_UPDRS_1_9"]
        + original_data["pre_UPDRS_1_10"]
        + original_data["pre_UPDRS_1_11"]
        + original_data["pre_UPDRS_1_12"]
        + original_data["pre_UPDRS_1_13"]
    )

    new_data["TOT_pre_UPDRS_II_ON"] = (
        original_data["pre_UPDRS_2_1"]
        + original_data["pre_UPDRS_2_2"]
        + original_data["pre_UPDRS_2_3"]
        + original_data["pre_UPDRS_2_4"]
        + original_data["pre_UPDRS_2_5"]
        + original_data["pre_UPDRS_2_6"]
        + original_data["pre_UPDRS_2_7"]
        + original_data["pre_UPDRS_2_8"]
        + original_data["pre_UPDRS_2_9"]
        + original_data["pre_UPDRS_2_10"]
        + original_data["pre_UPDRS_2_11"]
        + original_data["pre_UPDRS_2_12"]
        + original_data["pre_UPDRS_2_13"]
    )

    new_data["pre_BDI_SCORE"] = (
        original_data["pre_BDI_1"]
        + original_data["pre_BDI_2"]
        + original_data["pre_BDI_3"]
        + original_data["pre_BDI_4"]
        + original_data["pre_BDI_5"]
        + original_data["pre_BDI_6"]
        + original_data["pre_BDI_7"]
        + original_data["pre_BDI_8"]
        + original_data["pre_BDI_9"]
        + original_data["pre_BDI_10"]
        + original_data["pre_BDI_11"]
        + original_data["pre_BDI_12"]
        + original_data["pre_BDI_13"]
        + original_data["pre_BDI_14"]
        + original_data["pre_BDI_15"]
        + original_data["pre_BDI_16"]
        + original_data["pre_BDI_17"]
        + original_data["pre_BDI_18"]
        + original_data["pre_BDI_19"]
        + original_data["pre_BDI_20"]
        + original_data["pre_BDI_21"]
    )

    new_data["pre_AS_SCORE"] = (
        original_data["pre_AS_1"]
        + original_data["pre_AS_2"]
        + original_data["pre_AS_3"]
        + original_data["pre_AS_4"]
        + original_data["pre_AS_5"]
        + original_data["pre_AS_6"]
        + original_data["pre_AS_7"]
        + original_data["pre_AS_8"]
        + original_data["pre_AS_9"]
        + original_data["pre_AS_10"]
        + original_data["pre_AS_11"]
        + original_data["pre_AS_12"]
        + original_data["pre_AS_13"]
        + original_data["pre_AS_14"]
    )

    new_data["AVG_STROOP_TMTB"] = (
        original_data["TMTB_SCORE"] + original_data["STROOP_SCORE"]
    )

    new_data["PRE_PDQ_SUB_SCORE_MOB"] = (
        original_data["pre_PDQ39_1"]
        + original_data["pre_PDQ39_2"]
        + original_data["pre_PDQ39_3"]
        + original_data["pre_PDQ39_4"]
        + original_data["pre_PDQ39_5"]
        + original_data["pre_PDQ39_6"]
        + original_data["pre_PDQ39_7"]
        + original_data["pre_PDQ39_8"]
        + original_data["pre_PDQ39_9"]
        + original_data["pre_PDQ39_10"]
    )

    new_data["PRE_PDQ_SUB_SCORE_ADL"] = (
        original_data["pre_PDQ39_11"]
        + original_data["pre_PDQ39_12"]
        + original_data["pre_PDQ39_13"]
        + original_data["pre_PDQ39_14"]
        + original_data["pre_PDQ39_15"]
        + original_data["pre_PDQ39_16"]
    )

    new_data["PRE_PDQ_SUB_SCORE_EMO"] = (
        original_data["pre_PDQ39_17"]
        + original_data["pre_PDQ39_18"]
        + original_data["pre_PDQ39_19"]
        + original_data["pre_PDQ39_20"]
        + original_data["pre_PDQ39_21"]
        + original_data["pre_PDQ39_22"]
    )

    new_data["PRE_PDQ_SUB_SCORE_STI"] = (
        original_data["pre_PDQ39_23"]
        + original_data["pre_PDQ39_24"]
        + original_data["pre_PDQ39_25"]
        + original_data["pre_PDQ39_26"]
    )

    new_data["PRE_PDQ_SUB_SCORE_SOC"] = (
        original_data["pre_PDQ39_27"]
        + original_data["pre_PDQ39_28"]
        + original_data["pre_PDQ39_29"]
    )

    new_data["PRE_PDQ_SUB_SCORE_COG"] = (
        original_data["pre_PDQ39_30"]
        + original_data["pre_PDQ39_31"]
        + original_data["pre_PDQ39_32"]
        + original_data["pre_PDQ39_33"]
    )

    new_data["PRE_PDQ_SUB_SCORE_COM"] = (
        original_data["pre_PDQ39_34"]
        + original_data["pre_PDQ39_35"]
        + original_data["pre_PDQ39_36"]
    )

    new_data["PRE_PDQ_SUB_SCORE_BOD"] = (
        original_data["pre_PDQ39_37"]
        + original_data["pre_PDQ39_38"]
        + original_data["pre_PDQ39_39"]
    )

    new_data["PRE_PDQ_SCORE"] = (
        (
            (
                (
                    original_data["pre_PDQ39_1"]
                    + original_data["pre_PDQ39_2"]
                    + original_data["pre_PDQ39_3"]
                    + original_data["pre_PDQ39_4"]
                    + original_data["pre_PDQ39_5"]
                    + original_data["pre_PDQ39_6"]
                    + original_data["pre_PDQ39_7"]
                    + original_data["pre_PDQ39_8"]
                    + original_data["pre_PDQ39_9"]
                    + original_data["pre_PDQ39_10"]
                )
                / 40
            )
            + (
                (
                    original_data["pre_PDQ39_11"]
                    + original_data["pre_PDQ39_12"]
                    + original_data["pre_PDQ39_13"]
                    + original_data["pre_PDQ39_14"]
                    + original_data["pre_PDQ39_15"]
                    + original_data["pre_PDQ39_16"]
                )
                / 24
            )
            + (
                (
                    original_data["pre_PDQ39_17"]
                    + original_data["pre_PDQ39_18"]
                    + original_data["pre_PDQ39_19"]
                    + original_data["pre_PDQ39_20"]
                    + original_data["pre_PDQ39_21"]
                    + original_data["pre_PDQ39_22"]
                )
                / 24
            )
            + (
                (
                    original_data["pre_PDQ39_23"]
                    + original_data["pre_PDQ39_24"]
                    + original_data["pre_PDQ39_25"]
                    + original_data["pre_PDQ39_26"]
                )
                / 16
            )
            + (
                (
                    original_data["pre_PDQ39_27"]
                    + original_data["pre_PDQ39_28"]
                    + original_data["pre_PDQ39_29"]
                )
                / 12
            )
            + (
                (
                    original_data["pre_PDQ39_30"]
                    + original_data["pre_PDQ39_31"]
                    + original_data["pre_PDQ39_32"]
                    + original_data["pre_PDQ39_33"]
                )
                / 16
            )
            + (
                (
                    original_data["pre_PDQ39_34"]
                    + original_data["pre_PDQ39_35"]
                    + original_data["pre_PDQ39_36"]
                )
                / 12
            )
            + (
                (
                    original_data["pre_PDQ39_37"]
                    + original_data["pre_PDQ39_38"]
                    + original_data["pre_PDQ39_39"]
                )
                / 12
            )
        )
        / 8
    ) * 100

    new_data["PRE_UPDRS_III_OFF_TOTAL_LEFT"] = (
        original_data["UPDRS_3OFF_3_LA"]
        + original_data["UPDRS_3OFF_3_LB"]
        + original_data["UPDRS_3OFF_4_L"]
        + original_data["UPDRS_3OFF_5_L"]
        + original_data["UPDRS_3OFF_6_L"]
        + original_data["UPDRS_3OFF_7_L"]
        + original_data["UPDRS_3OFF_8_L"]
        + original_data["UPDRS_3OFF_15_L"]
        + original_data["UPDRS_3OFF_17_LA"]
        + original_data["UPDRS_3OFF_17_LB"]
    )
    new_data["PRE_UPDRS_III_OFF_TOTAL_RIGHT"] = (
        original_data["UPDRS_3OFF_3_RA"]
        + original_data["UPDRS_3OFF_3_RB"]
        + original_data["UPDRS_3OFF_4_R"]
        + original_data["UPDRS_3OFF_5_R"]
        + original_data["UPDRS_3OFF_6_R"]
        + original_data["UPDRS_3OFF_7_R"]
        + original_data["UPDRS_3OFF_8_R"]
        + original_data["UPDRS_3OFF_15_R"]
        + original_data["UPDRS_3OFF_17_RA"]
        + original_data["UPDRS_3OFF_17_RB"]
    )

    new_data["PRE_UPDRS_III_ON_TOTAL_LEFT"] = (
        original_data["UPDRS_3ON_3_LA"]
        + original_data["UPDRS_3ON_3_LB"]
        + original_data["UPDRS_3ON_4_L"]
        + original_data["UPDRS_3ON_5_L"]
        + original_data["UPDRS_3ON_6_L"]
        + original_data["UPDRS_3ON_7_L"]
        + original_data["UPDRS_3ON_8_L"]
        + original_data["UPDRS_3ON_15_L"]
        + original_data["UPDRS_3ON_17_LA"]
        + original_data["UPDRS_3ON_17_LB"]
    )
    new_data["PRE_UPDRS_III_ON_TOTAL_RIGHT"] = (
        original_data["UPDRS_3ON_3_RA"]
        + original_data["UPDRS_3ON_3_RB"]
        + original_data["UPDRS_3ON_4_R"]
        + original_data["UPDRS_3ON_5_R"]
        + original_data["UPDRS_3ON_6_R"]
        + original_data["UPDRS_3ON_7_R"]
        + original_data["UPDRS_3ON_8_R"]
        + original_data["UPDRS_3ON_15_R"]
        + original_data["UPDRS_3ON_17_RA"]
        + original_data["UPDRS_3ON_17_RB"]
    )

    new_data["UPDRS_III_DIFFERENCE_DOPA_LEFT"] = (
        new_data["PRE_UPDRS_III_OFF_TOTAL_LEFT"]
        - new_data["PRE_UPDRS_III_ON_TOTAL_LEFT"]
    )
    new_data["UPDRS_III_DIFFERENCE_DOPA_RIGHT"] = (
        new_data["PRE_UPDRS_III_OFF_TOTAL_RIGHT"]
        - new_data["PRE_UPDRS_III_ON_TOTAL_RIGHT"]
    )
    new_data["PERCENTAGE_UPDRS_III_DIFFERENCE_DOPA_LEFT"] = (
        (new_data["UPDRS_III_DIFFERENCE_DOPA_LEFT"])
        / new_data["PRE_UPDRS_III_OFF_TOTAL_LEFT"]
        * 100
    )
    new_data["PERCENTAGE_UPDRS_III_DIFFERENCE_DOPA_RIGHT"] = (
        (new_data["UPDRS_III_DIFFERENCE_DOPA_RIGHT"])
        / new_data["PRE_UPDRS_III_OFF_TOTAL_RIGHT"]
        * 100
    )

    new_data["TOT_UPDRS3_OFF"] = (
        original_data["UPDRS_3OFF_1"]
        + original_data["UPDRS_3OFF_2"]
        + original_data["UPDRS_3OFF_3_N"]
        + original_data["UPDRS_3OFF_3_RA"]
        + original_data["UPDRS_3OFF_3_LA"]
        + original_data["UPDRS_3OFF_3_RB"]
        + original_data["UPDRS_3OFF_3_LB"]
        + original_data["UPDRS_3OFF_4_R"]
        + original_data["UPDRS_3OFF_4_L"]
        + original_data["UPDRS_3OFF_5_R"]
        + original_data["UPDRS_3OFF_5_L"]
        + original_data["UPDRS_3OFF_6_R"]
        + original_data["UPDRS_3OFF_6_L"]
        + original_data["UPDRS_3OFF_7_R"]
        + original_data["UPDRS_3OFF_7_L"]
        + original_data["UPDRS_3OFF_8_R"]
        + original_data["UPDRS_3OFF_8_L"]
        + original_data["UPDRS_3OFF_9"]
        + original_data["UPDRS_3OFF_10"]
        + original_data["UPDRS_3OFF_11"]
        + original_data["UPDRS_3OFF_12"]
        + original_data["UPDRS_3OFF_13"]
        + original_data["UPDRS_3OFF_14"]
        + original_data["UPDRS_3OFF_15_R"]
        + original_data["UPDRS_3OFF_15_L"]
        + original_data["UPDRS_3OFF_16_R"]
        + original_data["UPDRS_3OFF_16_L"]
        + original_data["UPDRS_3OFF_17_RA"]
        + original_data["UPDRS_3OFF_17_LA"]
        + original_data["UPDRS_3OFF_17_RB"]
        + original_data["UPDRS_3OFF_17_LB"]
        + original_data["UPDRS_3OFF_17_LK"]
        + original_data["UPDRS_3OFF_18"]
    )

    new_data["BRADYK_RIGID_RIGHT_OFF"] = (
        original_data["UPDRS_3OFF_3_RA"]
        + original_data["UPDRS_3OFF_3_RB"]
        + original_data["UPDRS_3OFF_4_R"]
        + original_data["UPDRS_3OFF_5_R"]
        + original_data["UPDRS_3OFF_6_R"]
        + original_data["UPDRS_3OFF_7_R"]
        + original_data["UPDRS_3OFF_8_R"]
    )

    new_data["PERC_BRADY_RIGHT_OFF"] = (
        new_data["BRADYK_RIGID_RIGHT_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["BRADYK_RIGID_LEFT_OFF"] = (
        original_data["UPDRS_3OFF_3_LA"]
        + original_data["UPDRS_3OFF_3_LB"]
        + original_data["UPDRS_3OFF_4_L"]
        + original_data["UPDRS_3OFF_5_L"]
        + original_data["UPDRS_3OFF_6_L"]
        + original_data["UPDRS_3OFF_7_L"]
        + original_data["UPDRS_3OFF_8_L"]
    )

    new_data["PERC_BRADY_LEFT_OFF"] = (
        new_data["BRADYK_RIGID_LEFT_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["TOT_BRADYK_RIGID_OFF"] = (
        new_data["BRADYK_RIGID_RIGHT_OFF"] + new_data["BRADYK_RIGID_LEFT_OFF"]
    )

    new_data["PERC_TOT_BRADYK_RIGID_OFF"] = (
        new_data["TOT_BRADYK_RIGID_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["TREMOR_RIGHT_OFF"] = (
        original_data["UPDRS_3OFF_15_R"]
        + original_data["UPDRS_3OFF_16_R"]
        + original_data["UPDRS_3OFF_17_RA"]
        + original_data["UPDRS_3OFF_17_RB"]
    )

    new_data["PERC_TREMOR_RIGHT_OFF"] = (
        new_data["TREMOR_RIGHT_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["TREMOR_LEFT_OFF"] = (
        original_data["UPDRS_3OFF_15_L"]
        + original_data["UPDRS_3OFF_16_L"]
        + original_data["UPDRS_3OFF_17_LA"]
        + original_data["UPDRS_3OFF_17_LB"]
    )

    new_data["PERC_TREMOR_LEFT_OFF"] = (
        new_data["TREMOR_LEFT_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["TOT_TREMOR_OFF"] = (
        new_data["TREMOR_RIGHT_OFF"] + new_data["TREMOR_LEFT_OFF"]
    )

    new_data["PERC_TOT_TREMOR_OFF"] = (
        new_data["TOT_TREMOR_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["AXIAL_OFF"] = (
        original_data["UPDRS_3OFF_9"]
        + original_data["UPDRS_3OFF_10"]
        + original_data["UPDRS_3OFF_11"]
        + original_data["UPDRS_3OFF_12"]
        + original_data["UPDRS_3OFF_13"]
    )

    new_data["PERC_AXIAL_OFF"] = (
        new_data["AXIAL_OFF"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["TOT_UPDRS3_ON"] = (
        original_data["UPDRS_3ON_1"]
        + original_data["UPDRS_3ON_2"]
        + original_data["UPDRS_3ON_3_N"]
        + original_data["UPDRS_3ON_3_RA"]
        + original_data["UPDRS_3ON_3_LA"]
        + original_data["UPDRS_3ON_3_RB"]
        + original_data["UPDRS_3ON_3_LB"]
        + original_data["UPDRS_3ON_4_R"]
        + original_data["UPDRS_3ON_4_L"]
        + original_data["UPDRS_3ON_5_R"]
        + original_data["UPDRS_3ON_5_L"]
        + original_data["UPDRS_3ON_6_R"]
        + original_data["UPDRS_3ON_6_L"]
        + original_data["UPDRS_3ON_7_R"]
        + original_data["UPDRS_3ON_7_L"]
        + original_data["UPDRS_3ON_8_R"]
        + original_data["UPDRS_3ON_8_L"]
        + original_data["UPDRS_3ON_9"]
        + original_data["UPDRS_3ON_10"]
        + original_data["UPDRS_3ON_11"]
        + original_data["UPDRS_3ON_12"]
        + original_data["UPDRS_3ON_13"]
        + original_data["UPDRS_3ON_14"]
        + original_data["UPDRS_3ON_15_R"]
        + original_data["UPDRS_3ON_15_L"]
        + original_data["UPDRS_3ON_16_R"]
        + original_data["UPDRS_3ON_16_L"]
        + original_data["UPDRS_3ON_17_RA"]
        + original_data["UPDRS_3ON_17_LA"]
        + original_data["UPDRS_3ON_17_RB"]
        + original_data["UPDRS_3ON_17_LB"]
        + original_data["UPDRS_3ON_17_LK"]
        + original_data["UPDRS_3ON_18"]
    )

    new_data["BRADYK_RIGID_RIGHT_ON"] = (
        original_data["UPDRS_3ON_3_RA"]
        + original_data["UPDRS_3ON_3_RB"]
        + original_data["UPDRS_3ON_4_R"]
        + original_data["UPDRS_3ON_5_R"]
        + original_data["UPDRS_3ON_6_R"]
        + original_data["UPDRS_3ON_7_R"]
        + original_data["UPDRS_3ON_8_R"]
    )

    new_data["PERC_BRADY_RIGHT_ON"] = (
        new_data["BRADYK_RIGID_RIGHT_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["BRADYK_RIGID_LEFT_ON"] = (
        original_data["UPDRS_3ON_3_LA"]
        + original_data["UPDRS_3ON_3_LB"]
        + original_data["UPDRS_3ON_4_L"]
        + original_data["UPDRS_3ON_5_L"]
        + original_data["UPDRS_3ON_6_L"]
        + original_data["UPDRS_3ON_7_L"]
        + original_data["UPDRS_3ON_8_L"]
    )

    new_data["PERC_BRADY_LEFT_ON"] = (
        new_data["BRADYK_RIGID_LEFT_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["TOT_BRADYK_RIGID_ON"] = (
        new_data["BRADYK_RIGID_RIGHT_ON"] + new_data["BRADYK_RIGID_LEFT_ON"]
    )

    new_data["PERC_TOT_BRADYK_RIGID_ON"] = (
        new_data["TOT_BRADYK_RIGID_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["TREMOR_RIGHT_ON"] = (
        original_data["UPDRS_3ON_15_R"]
        + original_data["UPDRS_3ON_16_R"]
        + original_data["UPDRS_3ON_17_RA"]
        + original_data["UPDRS_3ON_17_RB"]
    )

    new_data["PERC_TREMOR_RIGHT_ON"] = (
        new_data["TREMOR_RIGHT_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["TREMOR_LEFT_ON"] = (
        original_data["UPDRS_3ON_15_L"]
        + original_data["UPDRS_3ON_16_L"]
        + original_data["UPDRS_3ON_17_LA"]
        + original_data["UPDRS_3ON_17_LB"]
    )

    new_data["PERC_TREMOR_LEFT_ON"] = (
        new_data["TREMOR_LEFT_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["TOT_TREMOR_ON"] = new_data["TREMOR_RIGHT_ON"] + new_data["TREMOR_LEFT_ON"]

    new_data["PERC_TOT_TREMOR_ON"] = (
        new_data["TOT_TREMOR_ON"] / new_data["TOT_UPDRS3_ON"]
    ) * 100

    new_data["AXIAL_ON"] = (
        original_data["UPDRS_3ON_9"]
        + original_data["UPDRS_3ON_10"]
        + original_data["UPDRS_3ON_11"]
        + original_data["UPDRS_3ON_12"]
        + original_data["UPDRS_3ON_13"]
    )

    new_data["PERC_AXIAL_ON"] = (new_data["AXIAL_ON"] / new_data["TOT_UPDRS3_ON"]) * 100

    new_data["TOT_DOPA"] = new_data["TOT_UPDRS3_OFF"] - new_data["TOT_UPDRS3_ON"]

    new_data["PERC_TOT_DOPA"] = (
        new_data["TOT_DOPA"] / new_data["TOT_UPDRS3_OFF"]
    ) * 100

    new_data["BRADYK_RIGID_RIGHT_DOPA"] = (
        new_data["BRADYK_RIGID_RIGHT_OFF"] - new_data["BRADYK_RIGID_RIGHT_ON"]
    )
    new_data["BRADYK_RIGID_LEFT_DOPA"] = (
        new_data["BRADYK_RIGID_LEFT_OFF"] - new_data["BRADYK_RIGID_LEFT_ON"]
    )

    new_data["PERC_BRADYK_RIGID_RIGHT_DOPA"] = (
        new_data["BRADYK_RIGID_RIGHT_DOPA"] / new_data["BRADYK_RIGID_RIGHT_OFF"]
    ) * 100

    new_data["PERC_BRADYK_RIGID_LEFT_DOPA"] = (
        new_data["BRADYK_RIGID_LEFT_DOPA"] / new_data["BRADYK_RIGID_LEFT_OFF"]
    ) * 100

    new_data["TOT_BRADYK_RIGID_DOPA"] = (
        new_data["BRADYK_RIGID_RIGHT_DOPA"] + new_data["BRADYK_RIGID_LEFT_DOPA"]
    )

    new_data["PERC_TOT_BRADYK_RIGID_DOPA"] = (
        new_data["TOT_BRADYK_RIGID_DOPA"] / new_data["TOT_BRADYK_RIGID_OFF"]
    ) * 100

    new_data["TREMOR_RIGHT_DOPA"] = (
        new_data["TREMOR_RIGHT_OFF"] - new_data["TREMOR_RIGHT_ON"]
    )

    new_data["TREMOR_LEFT_DOPA"] = (
        new_data["TREMOR_LEFT_OFF"] - new_data["TREMOR_LEFT_ON"]
    )

    new_data["PERC_TREMOR_RIGHT_DOPA"] = (
        new_data["TREMOR_RIGHT_DOPA"] / new_data["TREMOR_RIGHT_OFF"]
    ) * 100

    new_data["PERC_TREMOR_LEFT_DOPA"] = (
        new_data["TREMOR_LEFT_DOPA"] / new_data["TREMOR_LEFT_OFF"]
    ) * 100

    new_data["TOT_TREMOR_DOPA"] = (
        new_data["TREMOR_RIGHT_DOPA"] + new_data["TREMOR_LEFT_DOPA"]
    )
    new_data["PERC_TOT_TREMOR_DOPA"] = (
        new_data["TOT_TREMOR_DOPA"] / new_data["TOT_TREMOR_OFF"]
    ) * 100

    new_data["AXIAL_DOPA"] = new_data["AXIAL_OFF"] - new_data["AXIAL_ON"]
    new_data["PERC_AXIAL_DOPA"] = (new_data["AXIAL_DOPA"] / new_data["AXIAL_OFF"]) * 100

    new_data["POST_UPDRS_III_OFF_MED_ON_DBS_TOTAL_LEFT"] = (
        original_data["UPDRS_3MF_SN_3_LA"]
        + original_data["UPDRS_3MF_SN_3_LB"]
        + original_data["UPDRS_3MF_SN_4_L"]
        + original_data["UPDRS_3MF_SN_5_L"]
        + original_data["UPDRS_3MF_SN_6_L"]
        + original_data["UPDRS_3MF_SN_7_L"]
        + original_data["UPDRS_3MF_SN_8_L"]
        + original_data["UPDRS_3MF_SN_15_L"]
        + original_data["UPDRS_3MF_SN_17_LA"]
        + original_data["UPDRS_3MF_SN_17_LB"]
    )
    new_data["POST_UPDRS_III_OFF_MED_ON_DBS_TOTAL_RIGHT"] = (
        original_data["UPDRS_3MF_SN_3_RA"]
        + original_data["UPDRS_3MF_SN_3_RB"]
        + original_data["UPDRS_3MF_SN_4_R"]
        + original_data["UPDRS_3MF_SN_5_R"]
        + original_data["UPDRS_3MF_SN_6_R"]
        + original_data["UPDRS_3MF_SN_7_R"]
        + original_data["UPDRS_3MF_SN_8_R"]
        + original_data["UPDRS_3MF_SN_15_R"]
        + original_data["UPDRS_3MF_SN_17_RA"]
        + original_data["UPDRS_3MF_SN_17_RB"]
    )

    new_data["TOT_pre_UPDRS_IV"] = (
        original_data["pre_UPDRS_4_1"]
        + original_data["pre_UPDRS_4_2"]
        + original_data["pre_UPDRS_4_3"]
        + original_data["pre_UPDRS_4_4"]
        + original_data["pre_UPDRS_4_5"]
        + original_data["pre_UPDRS_4_6"]
    )

    new_data["pre_UPDRS_4_1C"] = (
        original_data["pre_UPDRS_4_1B"] / original_data["pre_UPDRS_4_1A"]
    ) * 100

    new_data["pre_UPDRS_4_3C"] = (
        original_data["pre_UPDRS_4_3B"] / original_data["pre_UPDRS_4_3A"]
    ) * 100

    new_data["pre_UPDRS_4_6C"] = (
        original_data["pre_UPDRS_4_6B"] / original_data["pre_UPDRS_4_6A"]
    ) * 100

    new_data["TOT_UPDRS3_MF_SN"] = (
        original_data["UPDRS_3MF_SN_1"]
        + original_data["UPDRS_3MF_SN_2"]
        + original_data["UPDRS_3MF_SN_3_N"]
        + original_data["UPDRS_3MF_SN_3_RA"]
        + original_data["UPDRS_3MF_SN_3_LA"]
        + original_data["UPDRS_3MF_SN_3_RB"]
        + original_data["UPDRS_3MF_SN_3_LB"]
        + original_data["UPDRS_3MF_SN_4_R"]
        + original_data["UPDRS_3MF_SN_4_L"]
        + original_data["UPDRS_3MF_SN_5_R"]
        + original_data["UPDRS_3MF_SN_5_L"]
        + original_data["UPDRS_3MF_SN_6_R"]
        + original_data["UPDRS_3MF_SN_6_L"]
        + original_data["UPDRS_3MF_SN_7_R"]
        + original_data["UPDRS_3MF_SN_7_L"]
        + original_data["UPDRS_3MF_SN_8_R"]
        + original_data["UPDRS_3MF_SN_8_L"]
        + original_data["UPDRS_3MF_SN_9"]
        + original_data["UPDRS_3MF_SN_10"]
        + original_data["UPDRS_3MF_SN_11"]
        + original_data["UPDRS_3MF_SN_12"]
        + original_data["UPDRS_3MF_SN_13"]
        + original_data["UPDRS_3MF_SN_14"]
        + original_data["UPDRS_3MF_SN_15_R"]
        + original_data["UPDRS_3MF_SN_15_L"]
        + original_data["UPDRS_3MF_SN_16_R"]
        + original_data["UPDRS_3MF_SN_16_L"]
        + original_data["UPDRS_3MF_SN_17_RA"]
        + original_data["UPDRS_3MF_SN_17_LA"]
        + original_data["UPDRS_3MF_SN_17_RB"]
        + original_data["UPDRS_3MF_SN_17_LB"]
        + original_data["UPDRS_3MF_SN_17_LK"]
        + original_data["UPDRS_3MF_SN_18"]
    )

    new_data["BRADYK_RIGID_RIGHT_MF_SN"] = (
        original_data["UPDRS_3MF_SN_3_RA"]
        + original_data["UPDRS_3MF_SN_3_RB"]
        + original_data["UPDRS_3MF_SN_4_R"]
        + original_data["UPDRS_3MF_SN_5_R"]
        + original_data["UPDRS_3MF_SN_6_R"]
        + original_data["UPDRS_3MF_SN_7_R"]
        + original_data["UPDRS_3MF_SN_8_R"]
    )

    new_data["PERC_BRADY_RIGHT_MF_SN"] = (
        new_data["BRADYK_RIGID_RIGHT_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["BRADYK_RIGID_LEFT_MF_SN"] = (
        original_data["UPDRS_3MF_SN_3_LA"]
        + original_data["UPDRS_3MF_SN_3_LB"]
        + original_data["UPDRS_3MF_SN_4_L"]
        + original_data["UPDRS_3MF_SN_5_L"]
        + original_data["UPDRS_3MF_SN_6_L"]
        + original_data["UPDRS_3MF_SN_7_L"]
        + original_data["UPDRS_3MF_SN_8_L"]
    )

    new_data["PERC_BRADY_LEFT_MF_SN"] = (
        new_data["BRADYK_RIGID_LEFT_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["TOT_BRADYK_RIGID_MF_SN"] = (
        new_data["BRADYK_RIGID_RIGHT_MF_SN"] + new_data["BRADYK_RIGID_LEFT_MF_SN"]
    )

    new_data["PERC_TOT_BRADYK_RIGID_MF_SN"] = (
        new_data["TOT_BRADYK_RIGID_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["TREMOR_RIGHT_MF_SN"] = (
        original_data["UPDRS_3MF_SN_15_R"]
        + original_data["UPDRS_3MF_SN_16_R"]
        + original_data["UPDRS_3MF_SN_17_RA"]
        + original_data["UPDRS_3MF_SN_17_RB"]
    )

    new_data["PERC_TREMOR_RIGHT_MF_SN"] = (
        new_data["TREMOR_RIGHT_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["TREMOR_LEFT_MF_SN"] = (
        original_data["UPDRS_3MF_SN_15_L"]
        + original_data["UPDRS_3MF_SN_16_L"]
        + original_data["UPDRS_3MF_SN_17_LA"]
        + original_data["UPDRS_3MF_SN_17_LB"]
    )

    new_data["PERC_TREMOR_LEFT_MF_SN"] = (
        new_data["TREMOR_LEFT_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["TOT_TREMOR_MF_SN"] = (
        new_data["TREMOR_RIGHT_MF_SN"] + new_data["TREMOR_LEFT_MF_SN"]
    )

    new_data["PERC_TOT_TREMOR_MF_SN"] = (
        new_data["TOT_TREMOR_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["AXIAL_MF_SN"] = (
        original_data["UPDRS_3MF_SN_9"]
        + original_data["UPDRS_3MF_SN_10"]
        + original_data["UPDRS_3MF_SN_11"]
        + original_data["UPDRS_3MF_SN_12"]
        + original_data["UPDRS_3MF_SN_13"]
    )

    new_data["PERC_AXIAL_MF_SN"] = (
        new_data["AXIAL_MF_SN"] / new_data["TOT_UPDRS3_MF_SN"]
    ) * 100

    new_data["post_UPDRS_4_1C"] = (
        original_data["post_UPDRS_4_1B"] / original_data["post_UPDRS_4_1A"]
    ) * 100

    new_data["post_UPDRS_4_3C"] = (
        original_data["post_UPDRS_4_3B"] / original_data["post_UPDRS_4_3A"]
    ) * 100

    new_data["post_UPDRS_4_6C"] = (
        original_data["post_UPDRS_4_6B"] / original_data["post_UPDRS_4_6A"]
    ) * 100

    new_data["TOT_post_UPDRS_I"] = (
        original_data["post_UPDRS_1_1"]
        + original_data["post_UPDRS_1_2"]
        + original_data["post_UPDRS_1_3"]
        + original_data["post_UPDRS_1_4"]
        + original_data["post_UPDRS_1_5"]
        + original_data["post_UPDRS_1_6"]
        + original_data["post_UPDRS_1_7"]
        + original_data["post_UPDRS_1_8"]
        + original_data["post_UPDRS_1_9"]
        + original_data["post_UPDRS_1_10"]
        + original_data["post_UPDRS_1_11"]
        + original_data["post_UPDRS_1_12"]
        + original_data["post_UPDRS_1_13"]
    )

    new_data["TOT_post_UPDRS_II_ON"] = (
        original_data["post_UPDRS_2_1"]
        + original_data["post_UPDRS_2_2"]
        + original_data["post_UPDRS_2_3"]
        + original_data["post_UPDRS_2_4"]
        + original_data["post_UPDRS_2_5"]
        + original_data["post_UPDRS_2_6"]
        + original_data["post_UPDRS_2_7"]
        + original_data["post_UPDRS_2_8"]
        + original_data["post_UPDRS_2_9"]
        + original_data["post_UPDRS_2_10"]
        + original_data["post_UPDRS_2_11"]
    )

    new_data["post_BDI_SCORE"] = (
        original_data["post_BDI_1"]
        + original_data["post_BDI_2"]
        + original_data["post_BDI_3"]
        + original_data["post_BDI_4"]
        + original_data["post_BDI_5"]
        + original_data["post_BDI_6"]
        + original_data["post_BDI_7"]
        + original_data["post_BDI_8"]
        + original_data["post_BDI_9"]
        + original_data["post_BDI_10"]
        + original_data["post_BDI_11"]
        + original_data["post_BDI_12"]
        + original_data["post_BDI_13"]
        + original_data["post_BDI_14"]
        + original_data["post_BDI_15"]
        + original_data["post_BDI_16"]
        + original_data["post_BDI_17"]
        + original_data["post_BDI_18"]
        + original_data["post_BDI_19"]
        + original_data["post_BDI_20"]
        + original_data["post_BDI_21"]
    )

    new_data["post_AS_SCORE"] = (
        original_data["post_AS_1"]
        + original_data["post_AS_2"]
        + original_data["post_AS_3"]
        + original_data["post_AS_4"]
        + original_data["post_AS_5"]
        + original_data["post_AS_6"]
        + original_data["post_AS_7"]
        + original_data["post_AS_8"]
        + original_data["post_AS_9"]
        + original_data["post_AS_10"]
        + original_data["post_AS_11"]
        + original_data["post_AS_12"]
        + original_data["post_AS_13"]
        + original_data["post_AS_14"]
    )

    new_data["POST_PDQ_SUB_SCORE_MOB"] = (
        original_data["post_PDQ39_1"]
        + original_data["post_PDQ39_2"]
        + original_data["post_PDQ39_3"]
        + original_data["post_PDQ39_4"]
        + original_data["post_PDQ39_5"]
        + original_data["post_PDQ39_6"]
        + original_data["post_PDQ39_7"]
        + original_data["post_PDQ39_8"]
        + original_data["post_PDQ39_9"]
        + original_data["post_PDQ39_10"]
    )

    new_data["POST_PDQ_SUB_SCORE_ADL"] = (
        original_data["post_PDQ39_11"]
        + original_data["post_PDQ39_12"]
        + original_data["post_PDQ39_13"]
        + original_data["post_PDQ39_14"]
        + original_data["post_PDQ39_15"]
        + original_data["post_PDQ39_16"]
    )

    new_data["POST_PDQ_SUB_SCORE_EMO"] = (
        original_data["post_PDQ39_17"]
        + original_data["post_PDQ39_18"]
        + original_data["post_PDQ39_19"]
        + original_data["post_PDQ39_20"]
        + original_data["post_PDQ39_21"]
        + original_data["post_PDQ39_22"]
    )

    new_data["POST_PDQ_SUB_SCORE_STI"] = (
        original_data["post_PDQ39_23"]
        + original_data["post_PDQ39_24"]
        + original_data["post_PDQ39_25"]
        + original_data["post_PDQ39_26"]
    )

    new_data["POST_PDQ_SUB_SCORE_SOC"] = (
        original_data["post_PDQ39_27"]
        + original_data["post_PDQ39_28"]
        + original_data["post_PDQ39_29"]
    )

    new_data["POST_PDQ_SUB_SCORE_COG"] = (
        original_data["post_PDQ39_30"]
        + original_data["post_PDQ39_31"]
        + original_data["post_PDQ39_32"]
        + original_data["post_PDQ39_33"]
    )

    new_data["POST_PDQ_SUB_SCORE_COM"] = (
        original_data["post_PDQ39_34"]
        + original_data["post_PDQ39_35"]
        + original_data["post_PDQ39_36"]
    )

    new_data["POST_PDQ_SUB_SCORE_BOD"] = (
        original_data["post_PDQ39_37"]
        + original_data["post_PDQ39_38"]
        + original_data["post_PDQ39_39"]
    )

    new_data["POST_PDQ_SCORE"] = (
        (
            (
                (
                    original_data["post_PDQ39_1"]
                    + original_data["post_PDQ39_2"]
                    + original_data["post_PDQ39_3"]
                    + original_data["post_PDQ39_4"]
                    + original_data["post_PDQ39_5"]
                    + original_data["post_PDQ39_6"]
                    + original_data["post_PDQ39_7"]
                    + original_data["post_PDQ39_8"]
                    + original_data["post_PDQ39_9"]
                    + original_data["post_PDQ39_10"]
                )
                / 40
            )
            + (
                (
                    original_data["post_PDQ39_11"]
                    + original_data["post_PDQ39_12"]
                    + original_data["post_PDQ39_13"]
                    + original_data["post_PDQ39_14"]
                    + original_data["post_PDQ39_15"]
                    + original_data["post_PDQ39_16"]
                )
                / 24
            )
            + (
                (
                    original_data["post_PDQ39_17"]
                    + original_data["post_PDQ39_18"]
                    + original_data["post_PDQ39_19"]
                    + original_data["post_PDQ39_20"]
                    + original_data["post_PDQ39_21"]
                    + original_data["post_PDQ39_22"]
                )
                / 24
            )
            + (
                (
                    original_data["post_PDQ39_23"]
                    + original_data["post_PDQ39_24"]
                    + original_data["post_PDQ39_25"]
                    + original_data["post_PDQ39_26"]
                )
                / 16
            )
            + (
                (
                    original_data["post_PDQ39_27"]
                    + original_data["post_PDQ39_28"]
                    + original_data["post_PDQ39_29"]
                )
                / 12
            )
            + (
                (
                    original_data["post_PDQ39_30"]
                    + original_data["post_PDQ39_31"]
                    + original_data["post_PDQ39_32"]
                    + original_data["post_PDQ39_33"]
                )
                / 16
            )
            + (
                (
                    original_data["post_PDQ39_34"]
                    + original_data["post_PDQ39_35"]
                    + original_data["post_PDQ39_36"]
                )
                / 12
            )
            + (
                (
                    original_data["post_PDQ39_37"]
                    + original_data["post_PDQ39_38"]
                    + original_data["post_PDQ39_39"]
                )
                / 12
            )
        )
        / 8
    ) * 100

    new_data["TOT_post_UPDRS_IV"] = (
        original_data["post_UPDRS_4_1"]
        + original_data["post_UPDRS_4_2"]
        + original_data["post_UPDRS_4_3"]
        + original_data["post_UPDRS_4_4"]
        + original_data["post_UPDRS_4_5"]
        + original_data["post_UPDRS_4_6"]
    )

    new_data[
        [
            "Participant Id",
            "SEX",
            "PRE_IMPULS_YN",
            "INVALID_SYMPTOM",
            "PRE_TOT_LED",
            "pre_HY_OFF",
            "pre_HY_ON",
            "ACTIVE_CP_IN_DL_STN_LEFT",
            "ACTIVE_CP_IN_DL_STN_RIGHT",
            "POST_TOT_LED",
            "post_HY",
            "neurosurgeon",
            "PARKINSONISM",
            "PARKINSONISM_TYPE",
            "DATA_USE_FOR_RESEARCH",
            "IMPROVE_YN"
        ]
    ] = original_data[
        [
            "Participant Id",
            "SEX",
            "PRE_IMPULS_YN",
            "INVALID_SYMPTOM",
            "PRE_TOT_LED",
            "pre_HY_OFF",
            "pre_HY_ON",
            "ACTIVE_CP_IN_DL_STN_LEFT",
            "ACTIVE_CP_IN_DL_STN_RIGHT",
            "POST_TOT_LED",
            "post_HY",
            "neurosurgeon",
            "PARKINSONISM",
            "PARKINSONISM_TYPE",
            "DATA_USE_FOR_RESEARCH",
            "IMPROVE_YN"
        ]
    ]

    active_contact_points = ['ACTIVE_CP_IN_DL_STN_LEFT', 'ACTIVE_CP_IN_DL_STN_RIGHT']
    # the columns in active_contact points contact the following values: NaN, YES and NO. Convert to nan, 1 and 0
    new_data[active_contact_points] = new_data[active_contact_points].replace({'YES': 1, 'Partly': 0.5, 'NO': 0, 'NaN': np.nan})
    new_data['ACTIVE_CPS_IN_DL_STN'] = (new_data['ACTIVE_CP_IN_DL_STN_LEFT'] + new_data['ACTIVE_CP_IN_DL_STN_RIGHT']) / 2

    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    column_dict = {
        'DATA_USE_FOR_RESEARCH': 'NO Permission data use for research',
        'SCREEN_DT': 'Screening date',
        'SURGERY_DT': 'Surgery date',
        'OPTIM_STIMULPAR_DT': 'Follow-up date',
        'BIRTH_YR': 'Birth year',
        'PARKINSON_YR': 'Year PD diagnosis',
        'Days between screening and surgery': 'Days between screening and surgery',
        'Days between surgery and follow up': 'Days between surgery and follow up',
        'Days between screening and follow up': 'Days between screening and follow up',
        'AGE': 'Age',
        'ziekteduur_calc': 'Disease duration',
        'TOT_pre_UPDRS_I': 'Total UPDRS-I score',
        'TOT_pre_UPDRS_II_ON': 'Total UPDRS-II ON score',
        'pre_BDI_SCORE': 'BDI score',
        'pre_AS_SCORE': 'AS score',
        'AVG_STROOP_TMTB': 'Average Stroop and TMT B score',
        'PRE_PDQ_SUB_SCORE_MOB': 'PDQ-39 mobility (subscore)',
        'PRE_PDQ_SUB_SCORE_ADL': 'PDQ-39 ADL (subscore)',
        'PRE_PDQ_SUB_SCORE_EMO': 'PDQ-39 emotional well-being (subscore)',
        'PRE_PDQ_SUB_SCORE_STI': 'PDQ-39 stigma (subscore)',
        'PRE_PDQ_SUB_SCORE_SOC': 'PDQ-39 social support (subscore)',
        'PRE_PDQ_SUB_SCORE_COG': 'PDQ-39 cognition (subscore)',
        'PRE_PDQ_SUB_SCORE_COM': 'PDQ-39 communication (subscore)',
        'PRE_PDQ_SUB_SCORE_BOD': 'PDQ-39 bodily discomfort (subscore)',
        'PRE_PDQ_SCORE': 'PDQ-39 score',
        'PRE_UPDRS_III_OFF_TOTAL_LEFT': 'Total UPDRS-III OFF score (left)',
        'PRE_UPDRS_III_OFF_TOTAL_RIGHT': 'Total UPDRS-III OFF score (right)',
        'PRE_UPDRS_III_ON_TOTAL_LEFT': 'Total UPDRS-III ON score (left)',
        'PRE_UPDRS_III_ON_TOTAL_RIGHT': 'Total UPDRS-III ON score (right)',
        'UPDRS_III_DIFFERENCE_DOPA_LEFT': 'UPDRS-III improvement after dopamine (left)',
        'UPDRS_III_DIFFERENCE_DOPA_RIGHT': 'UPDRS-III improvement after dopamine (right)',
        'PERCENTAGE_UPDRS_III_DIFFERENCE_DOPA_LEFT': '% UPDRS-III improvement after dopamine (left)',
        'PERCENTAGE_UPDRS_III_DIFFERENCE_DOPA_RIGHT': '% UPDRS-III improvement after dopamine (right)',
        'TOT_UPDRS3_OFF': 'Total UPDRS-III OFF score',
        'BRADYK_RIGID_RIGHT_OFF': 'Right bradykinesia + rigidity OFF score (UPDRS-III subscore)',
        'PERC_BRADY_RIGHT_OFF': '% Right bradykinesia + rigidity vs. total UPDRS-III OFF',
        'BRADYK_RIGID_LEFT_OFF': 'Left total bradykinesia + rigidity OFF score (UPDRS-III subscore)',
        'PERC_BRADY_LEFT_OFF': '% Left bradykinesia + rigidity vs. total UPDRS-III OFF',
        'TOT_BRADYK_RIGID_OFF': 'Total bradykinesia + rigidity OFF score (UPDRS-III subscore)',
        'PERC_TOT_BRADYK_RIGID_OFF': '% Total bradykinesia + rigidity OFF score vs. Total UPDRS-III OFF',
        'TREMOR_RIGHT_OFF': 'Right tremor OFF score (UPDRS-III subscore)',
        'PERC_TREMOR_RIGHT_OFF': '% Right tremor vs. total UPDRS-III OFF',
        'TREMOR_LEFT_OFF': 'Left tremor OFF score (UPDRS-III subscore)',
        'PERC_TREMOR_LEFT_OFF': '% Left tremor vs. total UPDRS-III OFF',
        'TOT_TREMOR_OFF': 'Total tremor OFF score (UPDRS-III subscore)',
        'PERC_TOT_TREMOR_OFF': '% Total tremor vs. total UPDRS-III OFF',
        'AXIAL_OFF': 'Axial OFF score (UPDRS-III subscore)',
        'PERC_AXIAL_OFF': '% Axial vs. total UPDRS-III OFF',
        'TOT_UPDRS3_ON': 'Total UPDRS-III ON score',
        'BRADYK_RIGID_RIGHT_ON': 'Right bradykinesia + rigidity ON score (UPDRS-III subscore)',
        'PERC_BRADY_RIGHT_ON': '% Right bradykinesia + rigidity vs. total UPDRS-III ON',
        'BRADYK_RIGID_LEFT_ON': 'Left bradykinesia + rigidity ON score (UPDRS-III subscore)',
        'PERC_BRADY_LEFT_ON': '% Left bradykinesia + rigidity vs. total UPDRS-III ON',
        'TOT_BRADYK_RIGID_ON': 'Total bradykinesia + rigidity ON score (UPDRS-III subscore)',
        'PERC_TOT_BRADYK_RIGID_ON': '% Total bradykinesia + rigidity ON score vs. Total UPDRS-III ON',
        'TREMOR_RIGHT_ON': 'Right tremor ON score (UPDRS-III subscore)',
        'PERC_TREMOR_RIGHT_ON': '% Right tremor vs. total UPDRS-III ON',
        'TREMOR_LEFT_ON': 'Left tremor ON score (UPDRS-III subscore)',
        'PERC_TREMOR_LEFT_ON': '% Left tremor vs. total UPDRS-III ON',
        'TOT_TREMOR_ON': 'Total tremor ON score (UPDRS-III subscore)',
        'PERC_TOT_TREMOR_ON': '% Total tremor vs. total UPDRS-III ON',
        'AXIAL_ON': 'Axial ON score (UPDRS-III subscore)',
        'PERC_AXIAL_ON': '% Axial vs. total UPDRS-III ON',
        'TOT_DOPA': 'UPDRS-III improvement after dopamine',
        'PERC_TOT_DOPA': '% UPDRS-III improvement after dopamine',
        'BRADYK_RIGID_RIGHT_DOPA': 'Right bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'BRADYK_RIGID_LEFT_DOPA': 'Left bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'PERC_BRADYK_RIGID_RIGHT_DOPA': '% Right bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'PERC_BRADYK_RIGID_LEFT_DOPA': '% Left bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'TOT_BRADYK_RIGID_DOPA': 'Total bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'PERC_TOT_BRADYK_RIGID_DOPA': '% Total bradykinesia + rigidity improvement after dopamine (UPDRS-III subscore)',
        'TREMOR_RIGHT_DOPA': 'Right tremor improvement after dopamine (UPDRS-III subscore)',
        'TREMOR_LEFT_DOPA': 'Left tremor improvement after dopamine (UPDRS-III subscore)',
        'PERC_TREMOR_RIGHT_DOPA': '% Right tremor improvement after dopamine (UPDRS-III subscore)',
        'PERC_TREMOR_LEFT_DOPA': '% Left tremor improvement after dopamine (UPDRS-III subscore)',
        'TOT_TREMOR_DOPA': 'Total tremor improvement after dopamine (UPDRS-III subscore)',
        'PERC_TOT_TREMOR_DOPA': '% Total tremor improvement after dopamine (UPDRS-III subscore)',
        'AXIAL_DOPA': 'Axial improvement after dopamine (UPDRS-III subscore)',
        'PERC_AXIAL_DOPA': '% Axial improvement after dopamine (UPDRS-III subscore)',
        'POST_UPDRS_III_OFF_MED_ON_DBS_TOTAL_LEFT': 'Postoperative total UPDRS-III OFF medication ON DBS score (left)',
        'POST_UPDRS_III_OFF_MED_ON_DBS_TOTAL_RIGHT': 'Postoperative total UPDRS-III OFF medication ON DBS score (right)',
        'TOT_pre_UPDRS_IV': 'Total preoperative UPDRS-IV score',
        'pre_UPDRS_4_1C': '% of waking day dyskinesias present',
        'pre_UPDRS_4_3C': '% of waking day OFF',
        'pre_UPDRS_4_6C': '% of OFF time with dystonia',
        'TOT_UPDRS3_MF_SN': 'Postoperative total UPDRS-III OFF medication ON DBS score',
        'BRADYK_RIGID_RIGHT_MF_SN': 'Postoperative right bradykinesia + rigidity OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_BRADY_RIGHT_MF_SN': '% Postoperative right bradykinesia + rigidity OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'BRADYK_RIGID_LEFT_MF_SN': 'Postoperative left bradykinesia + rigidity OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_BRADY_LEFT_MF_SN': '% Postoperative left bradykinesia + rigidity OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'TOT_BRADYK_RIGID_MF_SN': 'Postoperative total bradykinesia + rigidity OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_TOT_BRADYK_RIGID_MF_SN': '% Postoperative total bradykinesia + rigidity OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'TREMOR_RIGHT_MF_SN': 'Postoperative right tremor OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_TREMOR_RIGHT_MF_SN': '% Postoperative right tremor OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'TREMOR_LEFT_MF_SN': 'Postoperative left tremor OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_TREMOR_LEFT_MF_SN': '% Postoperative left tremor OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'TOT_TREMOR_MF_SN': 'Postoperative total tremor OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_TOT_TREMOR_MF_SN': '% Postoperative total tremor OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'AXIAL_MF_SN': 'Postoperative axial OFF medication ON DBS score (UPDRS-III subscore)',
        'PERC_AXIAL_MF_SN': '% Postoperative axial OFF medication ON DBS score vs. total UPDRS-III OFF medication ON DBS score',
        'post_UPDRS_4_1C': 'Postoperative % of waking day dyskinesias present',
        'post_UPDRS_4_3C': 'Postoperative % of waking day OFF',
        'post_UPDRS_4_6C': 'Postoperative % of OFF time with dystonia',
        'TOT_post_UPDRS_I': 'Postoperative total UPDRS-I score',
        'TOT_post_UPDRS_II_ON': 'Postoperative total UPDRS-II ON score',
        'post_BDI_SCORE': 'Postoperative BDI score',
        'post_AS_SCORE': 'Postoperative AS score',
        'POST_PDQ_SUB_SCORE_MOB': 'Postoperative PDQ-39 mobility (subscore)',
        'POST_PDQ_SUB_SCORE_ADL': 'Postoperative PDQ-39 ADL (subscore)',
        'POST_PDQ_SUB_SCORE_EMO': 'Postoperative PDQ-39 emotional well-being (subscore)',
        'POST_PDQ_SUB_SCORE_STI': 'Postoperative PDQ-39 stigma (subscore)',
        'POST_PDQ_SUB_SCORE_SOC': 'Postoperative PDQ-39 social support (subscore)',
        'POST_PDQ_SUB_SCORE_COG': 'Postoperative PDQ-39 cognition (subscore)',
        'POST_PDQ_SUB_SCORE_COM': 'Postoperative PDQ-39 communication (subscore)',
        'POST_PDQ_SUB_SCORE_BOD': 'Postoperative PDQ-39 bodily discomfort (subscore)',
        'POST_PDQ_SCORE': 'Postoperative PDQ-39 score',
        'TOT_post_UPDRS_IV': 'Postoperative total UPDRS-IV score',
        'Participant Id': 'Participant Id',
        'SEX': 'Sex',
        'PRE_IMPULS_YN': 'Impulse control disorder',
        'INVALID_SYMPTOM': 'Most invalidating symptom',
        'PRE_TOT_LED': 'Total Levodopa Equivalent Dose',
        'pre_HY_OFF': 'Hoehn and Yahr OFF',
        'pre_HY_ON': 'Hoehn and Yahr ON',
        'ACTIVE_CP_IN_DL_STN_LEFT': 'Active contactpoint in left STN',
        'ACTIVE_CP_IN_DL_STN_RIGHT': 'Active contactpoint in right STN',
        'ACTIVE_CPS_IN_DL_STN': 'Active contactpoints in STN',
        'POST_TOT_LED': 'Postoperative total Levodopa Equivalent Dose',
        'post_HY': 'Postoperative Hoen and Yahr',
        'neurosurgeon': 'Neurosurgeon',
        'PARKINSONISM': 'Parkinsonism yes/no',
        'PARKINSONISM_TYPE': 'Parkinsonism type',
        "IMPROVE_YN": "Meaningful patient reported improvement yes/no"
    }

    # rename columns according to dictionary
    new_data = new_data.rename(columns=column_dict)

    return new_data.copy()


if __name__ == "__main__":
    dataframe = get_dataframe()
    dataframe.to_excel("export.xlsx")

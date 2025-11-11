import polars as pl
from survey_kit_data.census.cps_asec import cps_asec
from survey_kit.utilities.compress import compress_df
from survey_kit.utilities.dataframe import (
    summary,
    columns_from_list
)
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.statistics import Statistics
from survey_kit import logger, config



# %%
logger.info("Download the CPS ASEC (using in-development survey-kit-data package)")
d_cps = cps_asec(2025)
for keyi, dfi in d_cps.items():
    d_cps[keyi] = dfi.select(pl.all().name.to_lowercase())
df_rep_weights = d_cps["replicate_weights"]
df_hhld = d_cps["hhld"]
df_person = d_cps["person"]


# %%
logger.info("Get the columns we need for household income")
df_hhld = df_hhld.select(
    columns_from_list(df_hhld,columns=["*seq","htotval","hrhtype"])
)

df_person = df_person.select(
    columns_from_list(df_person,columns=["*seq","pppos","hhdrel","prdtrace"])
)

# %%
logger.info("Merge the household and person data sets together")
df_joined = df_hhld.join(
    df_person,
    how="inner",
    left_on=["h_seq"],
    right_on=["ph_seq"]
)
df_joined = compress_df(
    df_joined.join(
        df_rep_weights,
        on=["h_seq","pppos"],
        how="inner"
    ),
    check_string=True
)

# %%
logger.info("Collect after the join to avoid repeat load/join at replicate calculation")
df_joined = df_joined.collect().lazy()

_ = summary(df_joined)



# %%
logger.info("Get the estimates")
replicates = Replicates(
    weight_stub="pwwgt",
    df=df_joined,
    bootstrap=False
)


# %%
logger.info("   For all households")
c_all_households = pl.col("hhdrel").eq(1) & pl.col("hrhtype").lt(9)
c_white_only = pl.col("prdtrace") == 1
sc = StatCalculator(
    df_joined.filter(c_all_households).collect().lazy(),
    statistics=Statistics(
        stats=["n","mean","p25","p50","p75"],
        columns="htotval",
        # quantile_interpolated=True
    ),
    replicates=replicates
)

# %%
logger.info("   For all households with white householder")
sc_white = StatCalculator(
    df_joined.filter(c_all_households & c_white_only).collect().lazy(),
    statistics=Statistics(
        stats=["n","mean","p25","p50","p75"],
        columns="htotval",
        # quantile_interpolated=True
    ),
    replicates=replicates
)

# %%
logger.info("Compare all households to households with white householder")
sc_all_white = sc.compare(sc_white,display=False)["ratio"]

sc_all_white.print()
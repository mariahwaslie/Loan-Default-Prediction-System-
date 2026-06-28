from pydantic import BaseModel, Field, ConfigDict


class LoanRequest(BaseModel):
    """
    Full 75-feature schema with default values.
    Generated directly from BASE_LOAN structure.
    """

    model_config = ConfigDict(extra="allow")

    loan_amnt: float = Field(default=12000.0)
    term: int = Field(default=36)
    int_rate: float = Field(default=13.5)

    grade: str = Field(default="C")
    sub_grade: str = Field(default="C3")
    emp_length: int = Field(default=5)

    home_ownership: str = Field(default="RENT")
    annual_inc: float = Field(default=65000.0)
    verification_status: str = Field(default="Verified")
    purpose: str = Field(default="debt_consolidation")

    zip_code: int = Field(default=60600)
    dti: float = Field(default=18.4)

    delinq_2yrs: int = Field(default=0)
    fico_range_low: int = Field(default=690)
    inq_last_6mths: int = Field(default=1)
    mths_since_last_delinq: int = Field(default=0)

    open_acc: int = Field(default=10)
    pub_rec: int = Field(default=0)
    revol_bal: int = Field(default=8000)
    revol_util: float = Field(default=55.0)

    total_acc: int = Field(default=24)
    initial_list_status: int = Field(default=1)
    collections_12_mths_ex_med: int = Field(default=0)

    verification_status_joint: str = Field(default="NONE")
    tot_coll_amt: int = Field(default=0)

    open_acc_6m: int = Field(default=1)
    open_act_il: int = Field(default=2)
    open_il_12m: int = Field(default=1)
    open_il_24m: int = Field(default=2)

    total_bal_il: int = Field(default=16000)
    il_util: float = Field(default=70.0)

    open_rv_12m: int = Field(default=2)
    open_rv_24m: int = Field(default=4)

    max_bal_bc: int = Field(default=3000)
    all_util: float = Field(default=62.0)

    total_rev_hi_lim: int = Field(default=15000)
    inq_fi: int = Field(default=1)
    total_cu_tl: int = Field(default=0)
    inq_last_12m: int = Field(default=2)

    acc_open_past_24mths: int = Field(default=4)
    avg_cur_bal: int = Field(default=9000)

    bc_open_to_buy: int = Field(default=5000)
    bc_util: float = Field(default=58.0)

    mo_sin_old_il_acct: int = Field(default=120)
    mo_sin_old_rev_tl_op: int = Field(default=160)
    mo_sin_rcnt_rev_tl_op: int = Field(default=8)

    mort_acc: int = Field(default=1)
    mths_since_recent_bc: int = Field(default=8)
    mths_since_recent_inq: int = Field(default=3)
    mths_since_recent_revol_delinq: int = Field(default=0)

    num_accts_ever_120_pd: int = Field(default=0)
    num_actv_bc_tl: int = Field(default=3)
    num_actv_rev_tl: int = Field(default=5)

    num_bc_sats: int = Field(default=4)
    num_bc_tl: int = Field(default=8)
    num_il_tl: int = Field(default=6)

    num_op_rev_tl: int = Field(default=7)
    num_rev_accts: int = Field(default=14)

    num_tl_90g_dpd_24m: int = Field(default=0)
    num_tl_op_past_12m: int = Field(default=2)

    pct_tl_nvr_dlq: float = Field(default=95.0)
    percent_bc_gt_75: float = Field(default=25.0)

    pub_rec_bankruptcies: int = Field(default=0)

    tot_hi_cred_lim: int = Field(default=180000)
    total_bal_ex_mort: int = Field(default=26000)
    total_bc_limit: int = Field(default=11000)
    total_il_high_credit_limit: int = Field(default=23000)

    issue_d_month: int = Field(default=6)
    latitude: float = Field(default=41.88)
    longitude: float = Field(default=-87.63)

    earliest_cr_month: int = Field(default=8)
    earliest_cr_year: int = Field(default=2003)

    emp_length_missing: int = Field(default=0)
    num_tl_120dpd_2m_missing: int = Field(default=0)

    issue_month_num: int = Field(default=48)
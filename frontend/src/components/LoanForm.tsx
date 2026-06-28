import { useState } from "react";
import api from "../api";
import { DEFAULT_LOAN } from "../defaults";
import "./LoanForm.css";

export default function LoanForm() {
  const [loan, setLoan] = useState(DEFAULT_LOAN);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  async function predictLoan() {
    setLoading(true);
    try {
      const res = await api.post("/predict", loan);
      setPrediction(res.data.prediction);
      setProbability(res.data.default_probability);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="loan-container">

      <div className="card">
        <div className="section-title">Loan Risk Inputs</div>

        {/* Loan Amount */}
        <div className="field">
          <div className="field-top">
            <span className="label">Loan Amount</span>
            <span className="value">${loan.loan_amnt}</span>
          </div>
          <input className="slider" type="range"
            min="1000" max="40000" step="500"
            value={loan.loan_amnt}
            onChange={(e) =>
              setLoan({ ...loan, loan_amnt: Number(e.target.value) })
            }
          />
        </div>

        {/* Interest Rate */}
        <div className="field">
          <div className="field-top">
            <span className="label">Interest Rate</span>
            <span className="value">{loan.int_rate}%</span>
          </div>
          <input className="slider" type="range"
            min="5" max="25" step="0.5"
            value={loan.int_rate}
            onChange={(e) =>
              setLoan({ ...loan, int_rate: Number(e.target.value) })
            }
          />
        </div>

        {/* Annual Income */}
        <div className="field">
          <div className="field-top">
            <span className="label">Annual Income</span>
            <span className="value">${loan.annual_inc}</span>
          </div>
          <input className="slider" type="range"
            min="20000" max="200000" step="1000"
            value={loan.annual_inc}
            onChange={(e) =>
              setLoan({ ...loan, annual_inc: Number(e.target.value) })
            }
          />
        </div>

        {/* DTI */}
        <div className="field">
          <div className="field-top">
            <span className="label">DTI Ratio</span>
            <span className="value">{loan.dti}</span>
          </div>
          <input className="slider" type="range"
            min="0" max="40" step="0.5"
            value={loan.dti}
            onChange={(e) =>
              setLoan({ ...loan, dti: Number(e.target.value) })
            }
          />
        </div>

        {/* FICO */}
        <div className="field">
          <div className="field-top">
            <span className="label">FICO Score</span>
            <span className="value">{loan.fico_range_low}</span>
          </div>
          <input className="slider" type="range"
            min="580" max="850"
            value={loan.fico_range_low}
            onChange={(e) =>
              setLoan({ ...loan, fico_range_low: Number(e.target.value) })
            }
          />
        </div>

        {/* Open Accounts */}
        <div className="field">
          <div className="field-top">
            <span className="label">Open Accounts</span>
            <span className="value">{loan.open_acc}</span>
          </div>
          <input className="slider" type="range"
            min="0" max="30"
            value={loan.open_acc}
            onChange={(e) =>
              setLoan({ ...loan, open_acc: Number(e.target.value) })
            }
          />
        </div>

        {/* Revol Util */}
        <div className="field">
          <div className="field-top">
            <span className="label">Revolving Utilization</span>
            <span className="value">{loan.revol_util}%</span>
          </div>
          <input className="slider" type="range"
            min="0" max="120"
            value={loan.revol_util}
            onChange={(e) =>
              setLoan({ ...loan, revol_util: Number(e.target.value) })
            }
          />
        </div>

        {/* Total Accounts */}
        <div className="field">
          <div className="field-top">
            <span className="label">Total Accounts</span>
            <span className="value">{loan.total_acc}</span>
          </div>
          <input className="slider" type="range"
            min="0" max="60"
            value={loan.total_acc}
            onChange={(e) =>
              setLoan({ ...loan, total_acc: Number(e.target.value) })
            }
          />
        </div>

        {/* BC Util */}
        <div className="field">
          <div className="field-top">
            <span className="label">Bankcard Utilization</span>
            <span className="value">{loan.bc_util}%</span>
          </div>
          <input className="slider" type="range"
            min="0" max="120"
            value={loan.bc_util}
            onChange={(e) =>
              setLoan({ ...loan, bc_util: Number(e.target.value) })
            }
          />
        </div>

        {/* Mort Accounts */}
        <div className="field">
          <div className="field-top">
            <span className="label">Mortgage Accounts</span>
            <span className="value">{loan.mort_acc}</span>
          </div>
          <input className="slider" type="range"
            min="0" max="10"
            value={loan.mort_acc}
            onChange={(e) =>
              setLoan({ ...loan, mort_acc: Number(e.target.value) })
            }
          />
        </div>

      </div>

      <button className="button" onClick={predictLoan} disabled={loading}>
        {loading ? "Predicting..." : "Predict Risk"}
      </button>

      {probability !== null && (
        <div className="result-card">
          <div className="section-title">Prediction</div>
          <div className="result-value">
            {(probability * 100).toFixed(2)}%
          </div>
          <div>
            {prediction === 1 ? "High Risk" : "Low Risk"}
          </div>
        </div>
      )}

    </div>
  );
}
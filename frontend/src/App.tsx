import "./App.css";
import LoanForm from "./components/LoanForm";

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Loan Default Risk Predictor</h1>
        <p>
          Interactive ML demo powered by FastAPI + React + Docker
        </p>
      </header>

      <main className="app-main">
        <LoanForm />
      </main>
    </div>
  );
}

export default App;
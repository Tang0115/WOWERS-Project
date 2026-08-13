import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell";
import NationalMap from "./views/NationalMap";
import StatePortfolio from "./views/StatePortfolio";
import PlantDetail from "./views/PlantDetail";
import Opportunities from "./views/Opportunities";
import Plants from "./views/Plants";
import Analytics from "./views/Analytics";
import Reports from "./views/Reports";

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<NationalMap />} />
        <Route path="/opportunities" element={<Opportunities />} />
        <Route path="/plants" element={<Plants />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/state/:code" element={<StatePortfolio />} />
        <Route path="/plant/:id" element={<PlantDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}

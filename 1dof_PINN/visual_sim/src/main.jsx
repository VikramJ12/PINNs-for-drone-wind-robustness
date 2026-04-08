import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ArmSim from "@/arm_sim.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ArmSim />
  </StrictMode>
);

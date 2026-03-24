import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ArmSim from "@visual_sim/arm_sim.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ArmSim />
  </StrictMode>
);

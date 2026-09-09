const NODE_COLORS = {
  Patient: "#e74c3c",
  Diagnosis: "#8e44ad",
  Symptom: "#f39c12",
  Exam: "#2980b9",
  ExamResult: "#3498db",
  Finding: "#16a085",
  Treatment: "#27ae60",
  Medication: "#2ecc71",
  Outcome: "#1abc9c",
  Measurement: "#7f8c8d",
  Value: "#95a5a6",
  Unit: "#bdc3c7",
  ReferenceRange: "#34495e",
  Interpretation: "#d35400",
  AnatomicalSite: "#c0392b",
  Status: "#9b59b6",
  Course: "#f1c40f",
};

const SVG_NS = "http://www.w3.org/2000/svg";
const caseSelect = document.querySelector("#case-select");
const reloadButton = document.querySelector("#reload-button");
const edgeLabelsCheckbox = document.querySelector("#edge-labels");
const statusElement = document.querySelector("#status");
const statsElement = document.querySelector("#stats");
const nodesLayer = document.querySelector("#nodes");
const edgesLayer = document.querySelector("#edges");
const edgeTextLayer = document.querySelector("#edge-text");
const legendElement = document.querySelector("#legend");

let currentGraph = null;

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, value);
  });
  return element;
}

function shortLabel(label) {
  return label.length > 22 ? `${label.slice(0, 20)}…` : label;
}

function calculatePositions(nodes) {
  const positions = new Map();
  const patient = nodes.find((node) => node.type === "Patient");
  const remaining = nodes.filter((node) => node !== patient);
  const centerX = 700;
  const centerY = 440;
  const radiusX = 570;
  const radiusY = 350;

  if (patient) {
    positions.set(patient.node_id, { x: centerX, y: centerY });
  }

  remaining.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(remaining.length, 1);
    positions.set(node.node_id, {
      x: centerX + radiusX * Math.cos(angle),
      y: centerY + radiusY * Math.sin(angle),
    });
  });

  return positions;
}

function renderLegend(nodes) {
  legendElement.replaceChildren();
  const types = [...new Set(nodes.map((node) => node.type))].sort();

  types.forEach((type) => {
    const item = document.createElement("span");
    item.className = "legend-item";

    const color = document.createElement("span");
    color.className = "legend-color";
    color.style.background = NODE_COLORS[type] || "#cccccc";

    item.append(color, document.createTextNode(type));
    legendElement.append(item);
  });
}

function renderGraph(graph) {
  nodesLayer.replaceChildren();
  edgesLayer.replaceChildren();
  edgeTextLayer.replaceChildren();

  const positions = calculatePositions(graph.nodes);
  const showEdgeLabels = edgeLabelsCheckbox.checked;

  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source_id);
    const target = positions.get(edge.target_id);
    if (!source || !target) return;

    edgesLayer.append(svgElement("line", {
      class: "edge",
      x1: source.x,
      y1: source.y,
      x2: target.x,
      y2: target.y,
    }));

    if (showEdgeLabels) {
      const label = svgElement("text", {
        class: "edge-label",
        x: (source.x + target.x) / 2,
        y: (source.y + target.y) / 2,
        "text-anchor": "middle",
      });
      label.textContent = edge.relation;
      edgeTextLayer.append(label);
    }
  });

  graph.nodes.forEach((node) => {
    const position = positions.get(node.node_id);
    const group = svgElement("g", {
      class: "node",
      transform: `translate(${position.x} ${position.y})`,
    });
    const radius = node.type === "Patient" ? 27 : 18;
    const circle = svgElement("circle", {
      r: radius,
      fill: NODE_COLORS[node.type] || "#cccccc",
    });
    const title = svgElement("title");
    title.textContent = `${node.label} · ${node.type}\n${node.node_id}`;
    circle.append(title);

    const label = svgElement("text", {
      x: 0,
      y: radius + 16,
      "text-anchor": "middle",
    });
    label.textContent = shortLabel(node.label);

    group.append(circle, label);
    nodesLayer.append(group);
  });

  renderLegend(graph.nodes);
  statsElement.textContent = `${graph.nodes.length} nós · ${graph.edges.length} relações`;
}

async function loadGraph() {
  const caseId = caseSelect.value;
  statusElement.className = "";
  statusElement.textContent = `Carregando ${caseId}...`;

  try {
    const response = await fetch(`/api/graph?case_id=${encodeURIComponent(caseId)}`);
    const graph = await response.json();
    if (!response.ok) throw new Error(graph.error || "Falha ao carregar o grafo.");

    currentGraph = graph;
    renderGraph(graph);
    statusElement.textContent = `Caso ${graph.case_id}`;
  } catch (error) {
    statusElement.className = "error";
    statusElement.textContent = error.message;
  }
}

async function loadCases() {
  try {
    const response = await fetch("/api/cases");
    const data = await response.json();
    if (!response.ok || !data.cases.length) {
      throw new Error("Nenhum caso encontrado.");
    }

    caseSelect.replaceChildren();
    data.cases.forEach((caseId) => {
      const option = document.createElement("option");
      option.value = caseId;
      option.textContent = caseId;
      caseSelect.append(option);
    });

    await loadGraph();
  } catch (error) {
    statusElement.className = "error";
    statusElement.textContent = error.message;
  }
}

reloadButton.addEventListener("click", loadGraph);
caseSelect.addEventListener("change", loadGraph);
edgeLabelsCheckbox.addEventListener("change", () => {
  if (currentGraph) renderGraph(currentGraph);
});

loadCases();

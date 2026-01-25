const conversations = [
  { id: "current", title: "Current chat", messages: [] }
];
let activeConversationId = "current";

function setStatus(el, type, text) {
  el.innerHTML = text ? `<span class="badge ${type}">${text}</span>` : "";
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text ?? "";
}

function setJson(id, data) {
  const el = document.getElementById(id);
  if (el) el.textContent = data ? JSON.stringify(data, null, 2) : "";
}

function clearIngest() {
  setText("ingestResult", "");
  setText("ingestKpiStatus", "-");
  setText("ingestKpiChunks", "-");
  setText("ingestKpiEntities", "-");
  setText("ingestKpiRelations", "-");
  setStatus(document.getElementById("ingestStatus"), "warn", "");
}

function clearQuery() {
  setText("queryResult", "");
  const chat = document.getElementById("chatLog");
  if (chat) chat.innerHTML = "";
  const convo = getActiveConversation();
  if (convo) convo.messages = [];
  setStatus(document.getElementById("queryStatus"), "warn", "");
}

function renderTools(tools) {
  const container = document.getElementById("toolBadges");
  if (!container) return;
  const toolList = [
    { key: "vector", label: "Vector" },
    { key: "graph", label: "Graph" },
    { key: "web", label: "Web" },
  ];
  container.innerHTML = "";
  toolList.forEach((t) => {
    const span = document.createElement("span");
    const on = !!tools[t.key];
    span.className = `tool ${on ? "on" : "off"}`;
    span.textContent = on ? `${t.label} ✓` : `${t.label} ✕`;
    container.appendChild(span);
  });
}

function appendMessage(role, text, meta) {
  const chat = document.getElementById("chatLog");
  if (!chat) return;
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const content = document.createElement("div");
  const cleanedText = role === "assistant" ? stripInlineCitations(text || "") : text || "";
  content.innerHTML = role === "assistant" ? renderMarkdown(cleanedText) : escapeHtml(cleanedText);
  bubble.appendChild(roleEl);
  bubble.appendChild(content);

  if (meta && role === "assistant") {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    const section = (title) => {
      const wrap = document.createElement("div");
      wrap.className = "meta-section";
      const label = document.createElement("div");
      label.className = "meta-title";
      label.textContent = title;
      wrap.appendChild(label);
      return { wrap, label };
    };

    // Knowledge source used
    const knowledgeSection = section("Knowledge source used");
    const provenance = meta.provenance || "";
    let provenanceLabel = "none";
    if (provenance) {
      provenanceLabel = provenance;
    } else {
      const hasInternal = !!meta.sources_used?.vector || !!meta.sources_used?.graph;
      const hasWeb = !!meta.sources_used?.web;
      if (hasInternal && hasWeb) provenanceLabel = "both";
      else if (hasInternal) provenanceLabel = "internal";
      else if (hasWeb) provenanceLabel = "online";
    }
    const provenanceLine = document.createElement("div");
    provenanceLine.className = "meta-line";
    provenanceLine.textContent = `Knowledge source used: ${provenanceLabel}`;
    knowledgeSection.wrap.appendChild(provenanceLine);
    metaEl.appendChild(knowledgeSection.wrap);

    // Sources used
    const sourcesSection = section("Sources used");
    const badges = document.createElement("div");
    badges.className = "tool-badges";
    ["vector", "graph", "web", "direct"].forEach((key) => {
      const span = document.createElement("span");
      const used = !!meta.sources_used?.[key];
      span.className = used ? "tool on" : "tool off";
      span.textContent = used ? `${key} ✓` : `${key} ✕`;
      badges.appendChild(span);
    });
    sourcesSection.wrap.appendChild(badges);
    metaEl.appendChild(sourcesSection.wrap);

    // Citations (internal + web)
    const citationsSection = section("Citations");
    const internalCitations = meta.citations || [];
    const webCitations = meta.web_citations || [];
    if (internalCitations.length || webCitations.length) {
      const ul = document.createElement("ul");
      ul.className = "meta-list";
      if (internalCitations.length) {
        const internalLabel = document.createElement("div");
        internalLabel.className = "meta-subtitle";
        internalLabel.textContent = "Internal citations";
        citationsSection.wrap.appendChild(internalLabel);
        internalCitations.forEach((c) => {
          const li = document.createElement("li");
          const fallback = (meta.retrieved_chunks || []).find(
            (rc) => rc.chunk_id === c.chunk_id
          );
          const name = c.doc_name || c.doc_id || "Unknown";
          const docLine = document.createElement("div");
          docLine.className = "meta-line";
          docLine.textContent = `Document: ${name}`;
          const pageLine = document.createElement("div");
          pageLine.className = "meta-line";
          pageLine.textContent = `Page: ${c.page_number ?? "?"}`;
          const chunkLine = document.createElement("div");
          chunkLine.className = "meta-line";
          chunkLine.textContent = `Chunk: ${c.chunk_id}`;
          const combined = c.similarity;
          if (combined !== null && combined !== undefined) {
            const scoreLine = document.createElement("div");
            scoreLine.className = "meta-line";
            scoreLine.textContent =
              `Hybrid score: ${combined.toFixed(3)} (semantic + keyword)`;
            li.appendChild(scoreLine);
          }
          li.appendChild(docLine);
          li.appendChild(pageLine);
          li.appendChild(chunkLine);
          ul.appendChild(li);
        });
      }

      if (webCitations.length) {
        const webLabel = document.createElement("div");
        webLabel.className = "meta-subtitle";
        webLabel.textContent = "Web citations";
        citationsSection.wrap.appendChild(webLabel);
        webCitations.forEach((c) => {
          const li = document.createElement("li");
          const a = document.createElement("a");
          a.href = c.url || "#";
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = c.title || c.url || c.web_id;
          li.appendChild(a);
          if (c.snippet) {
            const snippet = document.createElement("div");
            snippet.className = "meta-snippet";
            snippet.textContent = c.snippet;
            li.appendChild(snippet);
          }
          ul.appendChild(li);
        });
      }

      citationsSection.wrap.appendChild(ul);
      metaEl.appendChild(citationsSection.wrap);
    }

    const trace = meta.decision_trace || {};
    const traceSection = section("Decision trace (thresholds & signals)");
    const ulTrace = document.createElement("ul");
    ulTrace.className = "meta-list";
    const addTrace = (label, value) => {
      const li = document.createElement("li");
      const rendered =
        value === null || value === undefined
          ? "n/a"
          : typeof value === "object"
            ? JSON.stringify(value)
            : value;
      li.textContent = `${label}: ${rendered}`;
      ulTrace.appendChild(li);
    };
    if (!Object.keys(trace).length) {
      addTrace("vector_low_threshold", "n/a");
      addTrace("vector_high_threshold", "n/a");
      addTrace("vector_best_score", "n/a");
      addTrace("graph_confidence", "n/a");
      addTrace("graph_confidence_threshold", "n/a");
    }
    addTrace("vector_low_threshold", trace.vector_low_threshold);
    addTrace("vector_high_threshold", trace.vector_high_threshold);
    addTrace("vector_best_score", trace.vector_best_score);
    addTrace("graph_confidence", trace.graph_confidence);
    addTrace("graph_confidence_threshold", trace.graph_confidence_threshold);
    addTrace("graph_triggered", trace.graph_triggered);
    addTrace("web_triggered", trace.web_triggered);
    addTrace("web_trigger_reason", trace.web_trigger_reason);
    traceSection.wrap.appendChild(ulTrace);
    metaEl.appendChild(traceSection.wrap);
    bubble.appendChild(metaEl);
  }

  wrapper.appendChild(bubble);
  chat.appendChild(wrapper);
  chat.scrollTop = chat.scrollHeight;
}

function stripInlineCitations(text) {
  if (!text) return "";
  const lines = text.split("\n");
  const output = [];
  let skipping = false;
  for (const line of lines) {
    const trimmed = line.trim().toLowerCase();
    const startsCitation =
      trimmed.startsWith("citations") ||
      trimmed.startsWith("sources used") ||
      trimmed.startsWith("source:") ||
      trimmed.startsWith("source usage") ||
      trimmed.startsWith("knowledge source") ||
      trimmed.startsWith("internal citations") ||
      trimmed.startsWith("web citations") ||
      trimmed.startsWith("decision trace");
    if (startsCitation) {
      skipping = true;
      break;
    }
    if (!skipping) output.push(line);
  }
  return output.join("\n").trim();
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  if (!text) return "-";
  let html = escapeHtml(text);
  html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^\- (.*)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
  html = html.replace(/\n{2,}/g, "</div><div class=\"section\">");
  return `<div class="section">${html}</div>`;
}

function getActiveConversation() {
  return conversations.find((c) => c.id === activeConversationId);
}

function renderConversationsList() {
  const list = document.getElementById("memoryList");
  if (!list) return;
  list.innerHTML = "";
  conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = `sidebar-item ${c.id === activeConversationId ? "active" : ""}`;
    item.textContent = c.title;
    item.onclick = () => {
      activeConversationId = c.id;
      renderConversationsList();
      renderChatFromMemory();
    };
    list.appendChild(item);
  });
}

function renderChatFromMemory() {
  const chat = document.getElementById("chatLog");
  if (!chat) return;
  chat.innerHTML = "";
  const convo = getActiveConversation();
  if (!convo) return;
  convo.messages.forEach((msg) => {
    appendMessage(msg.role, msg.content, msg.meta);
  });
}

function newConversation() {
  const id = `chat_${Date.now()}`;
  conversations.unshift({ id, title: "New chat", messages: [] });
  activeConversationId = id;
  renderConversationsList();
  renderChatFromMemory();
}

async function ingest() {
  const status = document.getElementById("ingestStatus");
  const btn = document.getElementById("ingestBtn");
  setStatus(status, "warn", "Uploading...");
  btn.disabled = true;
  btn.textContent = "Uploading...";

  const fileInput = document.getElementById("pdfFile");
  if (!fileInput.files.length) {
    setStatus(status, "error", "Please choose a PDF file.");
    btn.disabled = false;
    btn.textContent = "Ingest";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const docId = document.getElementById("docId").value.trim();
  if (docId) formData.append("doc_id", docId);

  try {
    const res = await fetch("/ingest", { method: "POST", body: formData });
    const data = await res.json();
    setStatus(status, res.ok ? "success" : "error", res.ok ? "Ingestion complete" : "Ingestion failed");
    setJson("ingestResult", data);
    setText("ingestKpiStatus", data.status || "-");
    setText("ingestKpiChunks", String(data.chunks_created ?? "-"));
    setText("ingestKpiEntities", String(data.entities_extracted ?? "-"));
    setText("ingestKpiRelations", String(data.relationships_extracted ?? "-"));
  } catch (err) {
    setStatus(status, "error", "Ingestion error");
    setJson("ingestResult", { error: String(err) });
  }
  btn.disabled = false;
  btn.textContent = "Ingest";
}

async function query() {
  const status = document.getElementById("queryStatus");
  const btn = document.getElementById("queryBtn");
  setStatus(status, "warn", "Querying...");
  btn.disabled = true;
  btn.textContent = "Querying...";

  const question = document.getElementById("question").value.trim();
  if (!question) {
    setStatus(status, "error", "Please enter a question.");
    btn.disabled = false;
    btn.textContent = "Ask";
    return;
  }

  const convo = getActiveConversation();
  const history = (convo?.messages || []).map((m) => ({
    role: m.role,
    content: m.content
  }));
  const payload = {
    question,
    top_k: parseInt(document.getElementById("topK").value || "5", 10),
    use_graph_context: true,
    history
  };

  appendMessage("user", question);
  if (convo) {
    if (!convo.title || convo.title === "New chat" || convo.title === "Current chat") {
      convo.title = question.slice(0, 32) + (question.length > 32 ? "..." : "");
      renderConversationsList();
    }
    convo.messages.push({ role: "user", content: question });
  }
  document.getElementById("question").value = "";

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    setStatus(status, res.ok ? "success" : "error", res.ok ? "Answer ready" : "Query failed");
    appendMessage("assistant", data.answer || "", {
      sources_used: data.sources_used,
      citations: data.citations || [],
      web_citations: data.web_citations || [],
      decision_trace: data.decision_trace || {},
      retrieved_chunks: data.retrieved_chunks || [],
      provenance: data.provenance
    });
    if (convo) {
      convo.messages.push({
        role: "assistant",
        content: data.answer || "",
        meta: {
          sources_used: data.sources_used,
          citations: data.citations || [],
          web_citations: data.web_citations || [],
          decision_trace: data.decision_trace || {},
          retrieved_chunks: data.retrieved_chunks || [],
          provenance: data.provenance
        }
      });
    }
  } catch (err) {
    setStatus(status, "error", "Query error");
    setJson("queryResult", { error: String(err) });
  }
  btn.disabled = false;
  btn.textContent = "Ask";
}

window.query = query;
window.ingest = ingest;
window.clearQuery = clearQuery;
window.clearIngest = clearIngest;
window.newConversation = newConversation;

renderConversationsList();

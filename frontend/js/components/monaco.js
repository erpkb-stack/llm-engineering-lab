// Lazy Monaco Editor loader (CDN) with a graceful <textarea>/<pre> fallback.
const VER = "0.45.0";
const BASE = `https://cdn.jsdelivr.net/npm/monaco-editor@${VER}/min`;
let loading = null;

function load() {
  if (window.monaco) return Promise.resolve(window.monaco);
  if (loading) return loading;
  loading = new Promise((resolve, reject) => {
    window.MonacoEnvironment = {
      getWorkerUrl: () => URL.createObjectURL(new Blob([`self.MonacoEnvironment={baseUrl:"${BASE}/"};importScripts("${BASE}/vs/base/worker/workerMain.js");`], { type: "text/javascript" })),
    };
    const s = document.createElement("script");
    s.src = `${BASE}/vs/loader.js`;
    s.onload = () => {
      window.require.config({ paths: { vs: `${BASE}/vs` } });
      window.require(["vs/editor/editor.main"], () => resolve(window.monaco), reject);
    };
    s.onerror = reject;
    document.head.append(s);
    setTimeout(() => reject(new Error("Monaco load timeout")), 15000);
  });
  return loading;
}

/** Create an editor. Returns {getValue, setValue}. */
export async function editor(host, value, { language = "python", readOnly = false } = {}) {
  try {
    const monaco = await load();
    host.innerHTML = "";
    const dark = document.documentElement.dataset.bsTheme === "dark";
    const ed = monaco.editor.create(host, { value, language, readOnly, theme: dark ? "vs-dark" : "vs", minimap: { enabled: false }, fontSize: 12.5,
      automaticLayout: true, scrollBeyondLastLine: false, wordWrap: "on" });
    return { getValue: () => ed.getValue(), setValue: (v) => ed.setValue(v), monaco: true };
  } catch {
    host.innerHTML = "";
    const ta = document.createElement(readOnly ? "pre" : "textarea");
    ta.className = readOnly ? "code m-0 h-100" : "form-control mono h-100";
    if (readOnly) ta.textContent = value; else ta.value = value;
    host.append(ta);
    return { getValue: () => (readOnly ? ta.textContent : ta.value), setValue: (v) => (readOnly ? (ta.textContent = v) : (ta.value = v)), monaco: false };
  }
}

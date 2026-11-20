const form = document.getElementById("login-form");
const email = document.getElementById("email");
const pw = document.getElementById("password");
const err = document.getElementById("login-error");
const btn = document.getElementById("login-btn");

try { const t = localStorage.getItem("llmlab-theme"); if (t) document.documentElement.dataset.bsTheme = t; } catch {}

document.getElementById("toggle-pw").addEventListener("click", (e) => {
  const show = pw.type === "password";
  pw.type = show ? "text" : "password";
  e.currentTarget.textContent = show ? "Hide" : "Show";
  e.currentTarget.setAttribute("aria-label", show ? "Hide password" : "Show password");
});
// Demo credentials come from the server, which only publishes a read-only viewer account in production.
let demo = null;
const fillDemo = document.getElementById("fill-demo");
fillDemo.addEventListener("click", () => {
  if (!demo) return;
  email.value = demo.email;
  pw.value = demo.password;
});
fetch("/api/auth/demo", { credentials: "same-origin" })
  .then((r) => (r.ok ? r.json() : null))
  .then((d) => {
    if (!d?.enabled) return;
    demo = d;
    document.getElementById("demo-email").textContent = d.email;
    document.getElementById("demo-password").textContent = d.password;
    document.getElementById("demo-label").textContent = d.read_only ? "Public demo (read-only viewer)" : `Demo account (${d.role_label})`;
    document.getElementById("demo-note").textContent = d.read_only
      ? "Run any module in simulation mode. Saving, benchmarks, fine-tuning and settings need a full account."
      : d.role_users ? "Role demos (same password): engineer@, researcher@, viewer@llmlab.local" : "";
    fillDemo.textContent = d.read_only ? "Use public demo" : "Use demo account";
    fillDemo.classList.remove("d-none");
    document.getElementById("demo-info").classList.remove("d-none");
  })
  .catch(() => {});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  err.classList.add("d-none");
  form.classList.add("was-validated");
  if (!form.checkValidity()) return;
  btn.disabled = true;
  btn.textContent = "Signing in…";
  try {
    const r = await fetch("/api/auth/login", {
      method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
      body: JSON.stringify({ email: email.value.trim(), password: pw.value, remember_me: document.getElementById("remember").checked }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data?.error?.message || "Login failed");
    window.location.href = "/app";
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove("d-none");
  } finally {
    btn.disabled = false;
    btn.textContent = "Log in";
  }
});

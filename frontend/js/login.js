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
document.getElementById("fill-demo").addEventListener("click", () => {
  email.value = "demo@llmlab.local";
  pw.value = "Demo@12345";
});

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

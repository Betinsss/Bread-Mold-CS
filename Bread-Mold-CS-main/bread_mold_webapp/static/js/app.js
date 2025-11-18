const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  loading.classList.remove("hidden");
  result.classList.add("hidden");

  const formData = new FormData();
  formData.append("image", input.files[0]);

  const res = await fetch("/analyze", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  document.getElementById("risk").innerText = data.risk;
  document.getElementById("coverage").innerText = data.coverage;
  document.getElementById("action").innerText = data.action;
  document.getElementById("annotated").src = data.annotated;

  loading.classList.add("hidden");
  result.classList.remove("hidden");
});

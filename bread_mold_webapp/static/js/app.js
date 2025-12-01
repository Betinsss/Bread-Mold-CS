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
  
  const verdictElement = document.getElementById("verdict");
  verdictElement.innerText = data.verdict;
  // Remove any existing verdict classes
  verdictElement.classList.remove("healthy", "not-healthy");
  // Add appropriate class based on verdict
  if (data.verdict === "Healthy") {
    verdictElement.classList.add("healthy");
  } else if (data.verdict === "Not Healthy") {
    verdictElement.classList.add("not-healthy");
  }

  document.getElementById("annotated").src = data.annotated;

  loading.classList.add("hidden");
  result.classList.remove("hidden");
});

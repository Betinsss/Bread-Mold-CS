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

  // Summary section
  document.getElementById("risk").innerText = data.risk;
  document.getElementById("coverage").innerText = data.coverage;
  document.getElementById("bread-type").innerText = data.bread_type;
  document.getElementById("mold-type").innerText = data.mold_type;

  // Image
  document.getElementById("annotated").src = data.annotated;

  // Verdict section
  const verdictElement = document.getElementById("verdict");
  verdictElement.innerText = data.verdict;
  verdictElement.classList.remove("healthy", "not-healthy");
  if (data.verdict === "Healthy") {
    verdictElement.classList.add("healthy");
  } else {
    verdictElement.classList.add("not-healthy");
  }
  document.getElementById("storage-time").innerText = data.storage_time;
  document.getElementById("bread-age").innerText = data.bread_age;

  // Breakdown cards
  document.getElementById("risk-detail").innerText = data.risk;
  document.getElementById("coverage-detail").innerText = data.coverage;
  document.getElementById("bread-type-detail").innerText = data.bread_type;
  document.getElementById("bread-age-detail").innerText = data.bread_age;
  document.getElementById("mold-type-detail").innerText = data.mold_type;
  document.getElementById("storage-time-detail").innerText = data.storage_time;
  const verdictDetail = document.getElementById("verdict-detail");
  verdictDetail.innerText = data.verdict;
  verdictDetail.classList.remove("healthy", "not-healthy");
  if (data.verdict === "Healthy") {
    verdictDetail.classList.add("healthy");
  } else {
    verdictDetail.classList.add("not-healthy");
  }

  // Mold information section
  const moldInfoSection = document.getElementById("mold-info-section");
  if (data.mold_info) {
    moldInfoSection.classList.remove("hidden");
    document.getElementById("mold-name").innerText = data.mold_info.name;
    document.getElementById("mold-description").innerText = data.mold_info.description;
    document.getElementById("mold-health-risk").innerText = data.mold_info.health_risk;
    document.getElementById("mold-characteristics").innerText = data.mold_info.characteristics || "";
  } else {
    moldInfoSection.classList.add("hidden");
  }

  // Recommended actions
  document.getElementById("action").innerText = data.action;
  document.getElementById("storage-tips").innerText = data.storage_tips;

  loading.classList.add("hidden");
  result.classList.remove("hidden");
});

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

  // Call FastAPI backend
  const res = await fetch("http://localhost:8000/api/analyze", {
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
  if (data.mold_information && data.mold_information.trim() !== "" && data.mold_type !== "None") {
    moldInfoSection.classList.remove("hidden");
    document.getElementById("mold-name").innerText = data.mold_type;
    
    const desc = data.mold_information;
    const sentences = desc.split('. ');
    
    // First 2 sentences as description
    const description = sentences.slice(0, 2).join('. ') + (sentences.length > 2 ? '.' : '');
    document.getElementById("mold-description").innerText = description;
    
    // Extract health risk info (sentences with "risk", "toxic", "allergic", "respiratory")
    const healthSentences = sentences.filter(s => 
      s.toLowerCase().includes('risk') || 
      s.toLowerCase().includes('toxic') || 
      s.toLowerCase().includes('allergic') || 
      s.toLowerCase().includes('respiratory') ||
      s.toLowerCase().includes('infection')
    );
    document.getElementById("mold-health-risk").innerText = healthSentences.length > 0 
      ? healthSentences.join('. ') + '.'
      : "May cause allergic reactions and respiratory issues in sensitive individuals.";
    
    // Extract characteristics (sentences with "appears", "grows", "spreads", "conditions")
    const charSentences = sentences.filter(s => 
      s.toLowerCase().includes('appears') || 
      s.toLowerCase().includes('grows') || 
      s.toLowerCase().includes('spreads') ||
      s.toLowerCase().includes('conditions') ||
      s.toLowerCase().includes('fuzzy') ||
      s.toLowerCase().includes('spots')
    );
    document.getElementById("mold-characteristics").innerText = charSentences.length > 0
      ? charSentences.join('. ') + '.'
      : "Characteristics vary by environmental conditions and growth stage.";
  } else {
    moldInfoSection.classList.add("hidden");
  }

  // Recommended actions
  document.getElementById("action").innerText = data.action;
  // For storage tips, use a generic message or extract from bread type
  const breadType = data.bread_type.toLowerCase();
  let storageTips = "Store in a cool, dry place away from direct sunlight.";
  if (breadType.includes("white")) {
    storageTips = "Keep in original packaging at room temperature for 5-7 days or freeze for up to 3 months.";
  } else if (breadType.includes("whole wheat")) {
    storageTips = "Store in cool, dry place for 3-5 days or refrigerate to extend freshness.";
  } else if (breadType.includes("sourdough")) {
    storageTips = "Keep cut-side down on cutting board or in paper bag for 2-3 days.";
  } else if (breadType.includes("flat")) {
    storageTips = "Store in airtight container at room temperature for 2-3 days or refrigerate for up to 1 week.";
  }
  document.getElementById("storage-tips").innerText = storageTips;

  loading.classList.add("hidden");
  result.classList.remove("hidden");
});

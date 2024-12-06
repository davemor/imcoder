function displayBox(message) {
  const element = document.querySelector("#embeddings-output");
  element.classList.remove("hidden");
  element.querySelector("#embeddings").value = message;
}

async function sendImage(event) {
  event.preventDefault();

  console.log("Sending image...");

  const formData = new FormData(this);
  try {
    const res = await fetch("/encode", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      console.log("Success:", data);
      displayBox(data.encoded);
    } else {
      throw new Error(`Status: ${res.status}`);
    }
  } catch (error) {
    console.error(error);
    displayBox("Error: " + error);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.querySelector("#upload-form");
  uploadForm.onsubmit = sendImage;
});

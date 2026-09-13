// Tabs: any element with data-tabs holds a role="tablist" of buttons and the
// role="tabpanel" elements they control.
document.querySelectorAll("[data-tabs]").forEach((group) => {
  const tabs = Array.from(group.querySelectorAll('[role="tab"]'));

  function select(tab) {
    tabs.forEach((other) => {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));

    // Arrow keys move between tabs, the way native tab widgets behave
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (event.key === "Home") next = tabs[0];
      if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) {
        event.preventDefault();
        select(next);
        next.focus();
      }
    });
  });

  select(tabs.find((tab) => tab.getAttribute("aria-selected") === "true") || tabs[0]);
});

// Lightbox: clicking a figure opens it full size in a <dialog>, which closes
// on Escape by itself. Clicking outside the image or the close button also closes it.
const lightbox = document.createElement("dialog");
lightbox.className = "lightbox";
lightbox.innerHTML = '<button class="lightbox-close" aria-label="Close">×</button><img alt="" />';
document.body.append(lightbox);

const lightboxImage = lightbox.querySelector("img");
lightbox.querySelector(".lightbox-close").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

document.querySelectorAll("figure img").forEach((img) => {
  img.tabIndex = 0;

  function open() {
    lightboxImage.src = img.src;
    lightboxImage.alt = img.alt;
    lightbox.showModal();
  }

  img.addEventListener("click", open);
  img.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
});

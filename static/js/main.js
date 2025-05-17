document.addEventListener("DOMContentLoaded", function () {
    const forms = document.querySelectorAll("form");
    forms.forEach(function (form) {
      form.addEventListener("submit", function () {
        const spinner = document.getElementById("loadingOverlay");
        if (spinner) {
          spinner.classList.remove("hidden");
        }
      });
    });
});

  document.addEventListener("DOMContentLoaded", function () {
        const toggle = document.getElementById("dropdownToggle");
        const menu = document.getElementById("dropdownMenu");

        toggle.addEventListener("click", () => {
            menu.classList.toggle("hidden");
        });

        // Klik di luar dropdown untuk menutup
        window.addEventListener("click", function (e) {
            if (!toggle.contains(e.target) && !menu.contains(e.target)) {
                menu.classList.add("hidden");
            }
        });
    });
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['Bahnschrift', '"Arial Narrow"', '"Segoe UI"', "sans-serif"],
        body: ['"Microsoft YaHei UI"', '"Segoe UI"', "sans-serif"]
      },
      letterSpacing: {
        tactical: "0.24em"
      },
      boxShadow: {
        panel: "0 0 0 1px rgba(255,255,255,0.08), 0 20px 80px rgba(0,0,0,0.32)"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

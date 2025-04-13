// Pie Chart
const getChartOptions = () => {
    return {
      series: [52.8, 26.8, 20.4],
      colors: ["#1C64F2", "#16BDCA", "#9061F9"],
      chart: {
        height: 300,
        width: "100%",
        type: "pie",
      },
      stroke: {
        colors: ["white"],
        lineCap: "",
      },
      plotOptions: {
        pie: {
          labels: {
            show: true,
          },
          size: "100%",
          dataLabels: {
            offset: -25
          }
        },
      },
      labels: ["Positif", "Negatif", "Netral"],
      dataLabels: {
        enabled: true,
        style: {
          fontFamily: "Plus Jakarta Sans, sans-serif",
        },
      },
      legend: {
        position: "bottom",
        fontFamily: "Plus Jakarta Sans, sans-serif",
      },
      yaxis: {
        labels: {
          formatter: function (value) {
            return value + "%"
          },
        },
      },
      xaxis: {
        labels: {
          formatter: function (value) {
            return value  + "%"
          },
        },
        axisTicks: {
          show: false,
        },
        axisBorder: {
          show: false,
        },
      },
    }
}
  
if (document.getElementById("pie-chart") && typeof ApexCharts !== 'undefined') {
    const chart = new ApexCharts(document.getElementById("pie-chart"), getChartOptions());
    chart.render();
}  

// Line Chart
const options = {
    
    xaxis: {
      show: true,
      categories: ['2021', '2022', '2023', '2024', '2025'],
      labels: {
        show: true,
        style: {
          fontFamily: "Plus Jakarta Sans, sans-serif",
          cssClass: 'text-xs font-normal fill-gray-500 dark:fill-gray-400'
        }
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    yaxis: {
        min: 0,
        max: 200,
      show: true,
      labels: {
        show: true,
        style: {
          fontFamily: "Plus Jakarta Sans, sans-serif",
          cssClass: 'text-xs font-normal fill-gray-500 dark:fill-gray-400'
        },
        formatter: function (value) {
          return value;
        }
      }
    },
    series: [
      {
        name: "Positif",
        data: [150, 141, 145, 152, 135],
        color: "#1A56DB",
      },
      {
        name: "Negatif",
        data: [43, 13, 65, 12, 42],
        color: "#7E3BF2",
      },
      {
        name: "Netral",
        data: [23, 43, 12, 65, 32],
        color: "#FBBF24",
      },
    ],
    chart: {
      sparkline: {
        enabled: false
      },
      height: "100%",
      width: "100%",
      type: "area",
      fontFamily: "Plus Jakarta Sans, sans-serif",
      dropShadow: {
        enabled: false,
      },
      toolbar: {
        show: false,
      },
    },
    tooltip: {
      enabled: true,
      x: {
        show: false,
      },
    },
    fill: {
      type: "gradient",
      gradient: {
        opacityFrom: 0.55,
        opacityTo: 0,
        shade: "#1C64F2",
        gradientToColors: ["#1C64F2"],
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      width: 6,
    },
    legend: {
      show: false
    },
    grid: {
      show: true,
    },
    }
    
    if (document.getElementById("labels-chart") && typeof ApexCharts !== 'undefined') {
    const chart = new ApexCharts(document.getElementById("labels-chart"), options);
    chart.render();
    }
    
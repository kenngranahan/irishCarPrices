"use strict";

const test = fetch("http://127.0.0.1:5500/price_data.json");

let bla = test
  .then((response) => response.json())
  .then((data) => {
    let year = data[4];
    let price = data[7];
    var trace1 = {
      x: year,
      y: price,
      mode: "markers",
      type: "scatter",
    };
    var data = [trace1];
    Plotly.newPlot("myDiv", data);
  });

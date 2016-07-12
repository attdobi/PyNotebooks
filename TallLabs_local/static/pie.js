// The base endpoint to receive data from. See update_url()
var URL_BASE = "http://localhost:5000/";

// Update graph in response to inputs
d3.select("#word").on("input", make_graph);
//d3.select("#day_select").on("input", make_graph);
//d3.select("#station_select").on("input", make_graph);
//d3.select("#time").on("input", make_graph);

var margin = {top: 20, right: 20, bottom: 100, left: 60};
var width = 600 - margin.left - margin.right;
var height = 400 - margin.top - margin.bottom;

// Whitespace on either side of the bars in units of minutes
var binMargin = .1;

var x = d3.scale.linear()
    .range([0,  width]);
    //.domain([0, 25]);
var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");
    //.ticks(6);
var y = d3.scale.linear()
    .range([height, 0]);
var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .ticks(10);

var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
      "translate(" + margin.left + "," + margin.top + ")");

// x axis
svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .text("emoji")
      .attr("dy", "3em")
      .attr("text-align", "center")
      .attr("x", width / 2 - margin.right - margin.left);

// y axis
svg.append("g")
    .attr("class", "y axis")
    .call(yAxis)
  .append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("dy", "-2em")
    .text("Count");

/* Update the time displayed (XX:XX) next to the time slider
function update_slider(time) {
  var dateObj = new Date();
  dateObj.setHours(Math.floor(time/60));
  dateObj.setMinutes(time % 60);
  d3.select("#prettyTime")
    .text(dateObj.toTimeString().substring(0, 5));
}
*/

// Return url to recieve csv data with query filled in from input fields
function update_url() {
  return URL_BASE +"?word=" + document.getElementById("word").value;
        //"&time=" + document.getElementById("time").value +
        //"&station=" + document.getElementById("station_select").value +
        //"&day=" + document.getElementById("day_select").value;
}

// Convert csv data to number types
function type(d) {
window.alert(d)
  //d.emoji = +d.emoji;
  d.freqency = +d.freqency;
  return d;
}

function make_graph() {
  //update_slider(+document.getElementById("time").value);
  url = update_url()
//  d3.csv(url, type, function(error, data) {
 //   y.domain([0, d3.max(data, function(d) { return d.freqency; })]);

// Get the data
d3.csv(url, function(error, data) {
	data.forEach(function(d) {
		d.emoji = d.emoji;
		d.freqency = +d.freqency;
	});

    bars.transition(1000)
      .attr("y", function(d) { return  y(d.freqency); } )
      .attr("height", function(d) { return height - y(d.freqency); } );

    bars.enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d) { return x(d.emoji); })
      .attr("width", x(1 - 2 * binMargin))
      .attr("y", height)
      .attr("height", 0)
      .transition(1000)
        .attr("y", function(d) { return y(d.freqency); })
        .attr("height", function(d) { return height - y(d.freqency); });

    bars.exit()
      .transition(1000)
        .attr("y", height)
        .attr("height", 0)
      .remove();
  });
}

make_graph();
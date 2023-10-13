/**
 *
 * This demo is part of Burn project: https://github.com/burn-rs/burn
 * 
 * Released under a dual license: 
 * https://github.com/burn-rs/burn/blob/main/LICENSE-MIT
 * https://github.com/burn-rs/burn/blob/main/LICENSE-APACHE
 * 
 */


/**
 * Looks up element by an id.
 * @param {string} - Element id.
 */
function $(id) {
    return document.getElementById(id);
}

/**
 * Truncates number to a given decimal position
 * @param {number} num - Number to truncate.
 * @param {number} fixed - Decimal positions.
 * src: https://stackoverflow.com/a/11818658
 */
function toFixed(num, fixed) {
    const re = new RegExp('^-?\\d+(?:\.\\d{0,' + (fixed || -1) + '})?');
    return num.toString().match(re)[0];
}

/**
 * Helper function that builds a chart using Chart.js library.
 * @param {object} chartEl - Chart canvas element.
 * 
 * NOTE: Assumes chart.js is loaded into the global.
 */
function chartConfigBuilder(chartEl) {
    Chart.register(ChartDataLabels);
    return new Chart(chartEl, {
        plugins: [ChartDataLabels],
        type: "bar",
        data: {
            labels: ["", "", "", "", "",],
            datasets: [
                {
                    data: [0.0, 0.0, 0.0, 0.0, 0.0], // Added one more data point to make it 10
                    borderWidth: 0,
                    fill: true,
                    backgroundColor: "#247ABF",
                    axis: 'y',
                },
            ],
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: true,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: true,
                },
                datalabels: {
                    color: "white",
                    formatter: function (value, context) {
                        return toFixed(value, 2);
                    },
                },
            },
            indexAxis: 'y',
            scales: {
                y: {
                },
                x: {
                    suggestedMin: 0.0,
                    suggestedMax: 1.0,
                    beginAtZero: true,

                },
            },
        },
    });
}

/** Helper function that extracts labels and probabilities from the data.
 * @param {object} data - Data object.
 * @returns {object} - Object with labels and probabilities.
 */
function extractLabelsAndProbabilities(data) {
    const labels = [];
    const probabilities = [];

    for (let item of data) {
        if (item.hasOwnProperty('label') && item.hasOwnProperty('probability')) {
            labels.push(item.label);
            probabilities.push(item.probability);
        }
    }

    return {
        labels,
        probabilities
    };
}

/**
 * Helper function that extracts RGB values from a canvas.
 * @param {object} canvas - Canvas element.
 * @param {object} ctx - Canvas context.
 * @returns {object} - Flattened array of RGB values.
 */
function extractRGBValuesFromCanvas(canvas, ctx) {
    // Get image data from the canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Get canvas dimensions
    const height = canvas.height;
    const width = canvas.width;

    // Create a flattened array to hold the RGB values in channel-first order
    const flattenedArray = new Float32Array(3 * height * width);

    // Initialize indices for R, G, B channels in the flattened array
    let kR = 0,
        kG = height * width,
        kB = 2 * height * width;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Compute the index for the image data array
            const index = (y * width + x) * 4;

            // Fill in the R, G, B channels in the flattened array
            flattenedArray[kR++] = imageData.data[index] / 255.0; // Red
            flattenedArray[kG++] = imageData.data[index + 1] / 255.0; // Green
            flattenedArray[kB++] = imageData.data[index + 2] / 255.0; // Blue
        }
    }

    return flattenedArray;
}

/** Detect if browser is safari
 * @returns {boolean} - True if browser is safari.
 */
function isSafari() {
    // https://stackoverflow.com/questions/7944460/detect-safari-browser
    let isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    return isSafari;
}
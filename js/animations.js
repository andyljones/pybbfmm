import d3 from 'https://d3js.org/d3.v5.min.js';

function random_problem(S, T, W) {
    return {
        sources: d3.range(0, S).map(() => d3.randomUniform(.5, W-.5)),
        targets: d3.range(0, W, W/T),
        kernel: (a, b) => 1/(1 + ((a-b)*2)**2)
    }
}
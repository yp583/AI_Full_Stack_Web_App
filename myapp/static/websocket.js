const socket = new WebSocket('ws://localhost:8000/ws/livec/1/');
socket.onmessage = (e) => {
    result = JSON.parse(e.data);
    if (result.hasOwnProperty("cost")){
        document.getElementById("training_txt").innerHTML = "Cost: " + result.cost;
    }
    if (result.hasOwnProperty("colors")){
        for (let i = 0; i < result.colors.length; i++) {
            if (result.colors[i][0] == 1){
                document.getElementById("graph-datapoint-" + i).style.backgroundColor = "#FF0000";
            }
            else{
                document.getElementById("graph-datapoint-" + i).style.backgroundColor = "#0000FF";
            }
        }
    }
    if (result.hasOwnProperty("newdata")){ 
        for (let i = 0; i < result.newdata.length; i++) {
            document.getElementById("graph-datapoint-" + i).style.right = result.newdata[i][0].toString() + "%";
            document.getElementById("graph-datapoint-" + i).style.top = result.newdata[i][1].toString() + "%";
            document.getElementById("graph-datapoint-" + i).style.backgroundColor = result.newdata[i][2];
        }
    }
}

socket.onclose = (e) => {
    console.log("Socket closed!");
}
function starttraining() {
    epoch_num = document.querySelector("#epoch").value;
    batch_size = document.querySelector("#batch_size").value;
    random_seed = document.querySelector("#random_seed").value;
    socket.send(JSON.stringify(
        {
            train: "true",
            epochs: parseInt(epoch_num),
            batch_size: parseInt(batch_size),
            random_seed: random_seed
        }
    ));
}
function send_data_gen() {
    noise = document.querySelector("#noise").value;
    random_seed = document.querySelector("#random_seed").value;
    socket.send(JSON.stringify(
        {
            noise: noise,
            random_seed: random_seed
        }
    ));
}
function reset() {
    socket.send(JSON.stringify(
        {
            reset: "true"
        }
    ));
}

function draw_circle_canvas(ctx, x, y, r){
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
}
function draw_line_canvas(ctx, x1, y1, x2, y2){
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}
let currLayers = 2;
let nodeNums = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

function draw_layers_canvas(layers){
    const canvas = document.getElementById("myCanvas");
    const width = canvas.width;
    const height = canvas.height - canvas.height/5;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, canvas.height);

    let prevx = 0;
    let prevy = [];
    let curr_y = []

    for (let i = 0; i < layers; i++) {
        for (let n = 0; n < nodeNums[i]; n++) {
            let x = Math.floor((i * width/layers) + width/(2 * layers));
            let y = Math.floor((n * height/nodeNums[i]) + height/(2 * nodeNums[i]) + canvas.height/10);

            curr_y.push(y);
            
            draw_circle_canvas(ctx, x, y, Math.floor(height/25));
            if (i > 0){
                for (let pn = 0; pn < nodeNums[i-1]; pn++){
                    draw_line_canvas(ctx, prevx, prevy[pn], x - height/25, y);
                }
            }
            else if (i == 0){
                prevx = x;
                prevy.push(y);
            }
        }
        prevx = Math.floor((i * width/layers) + width/(2 * layers)) + height/25;
        prevy = curr_y;
        curr_y = [];
    }
    currLayers = layers;
}
function add_node(layer){
    nodeNums[layer] = Math.min(10, nodeNums[layer] + 1);
    update_layers(currLayers);
}
function delete_node(layer){
    if (nodeNums[layer] == 1 && currLayers > 2){
        currLayers -= 1;
    }
    nodeNums[layer] = Math.max(1, nodeNums[layer] - 1);
    update_layers(currLayers);
}

function update_layers(layers){
    const canvas = document.getElementById("myCanvas");
    const width = canvas.width;
    const height = canvas.height;
    currLayers = layers;
    for (let i = 0; i < 10; i++) {
        if (i < layers){
            nodeNums[i] = Math.max(1, nodeNums[i]);
            document.getElementById("add-node-btn-" + i).style = "position: absolute; display:block; left: " + (Math.floor((i * width/layers) + width/(2 * layers)) - 20) + "px";
            document.getElementById("del-node-btn-" + i).style = "position: absolute; display:block; bottom: " + -height + "px; left: " + (Math.floor((i * width/layers) + width/(2 * layers)) - 20) + "px";
        }
        else{
            document.getElementById("add-node-btn-" + i).style = "position: absolute; display:none;";
            document.getElementById("del-node-btn-" + i).style = "position: absolute; display:none;";
        }
    }
    draw_layers_canvas(layers);
    document.getElementById("layers_num").value = currLayers;
    document.getElementById("layers_num_txt").value = "Number of Layers: " + currLayers;
}

function canvas_init(){
    const canvas = document.getElementById("myCanvas");
    const ctx = canvas.getContext("2d");
    ctx.canvas.width  = window.innerWidth-20;
    ctx.canvas.height = Math.floor(0.7 * window.innerHeight);
}
function canvas_resize(){
    const canvas = document.getElementById("myCanvas");
    const ctx = canvas.getContext("2d");
    ctx.canvas.width  = window.innerWidth-20;
    ctx.canvas.height = Math.floor(0.7 * window.innerHeight);
    update_layers(currLayers);
}

canvas_init()
update_layers(2);
window.onresize = canvas_resize;


let tab = 1;
function switchtab(tab_to_switch){
    if (tab != tab_to_switch){
        document.getElementById("tab-btn-" + tab_to_switch.toString()).style = "padding: 3px 12%; margin: 0px 0.7%";
        document.getElementById("tab-btn-" + tab_to_switch.toString()).className = "btn btn-info";

        if (document.getElementById("tab-content-" + tab_to_switch.toString()) != null){
            document.getElementById("tab-content-" + tab_to_switch.toString()).style = "display: block";
        }
        for (let i = 1; i <= 2; i++) {
            if (i != tab_to_switch){
                document.getElementById("tab-btn-" + i.toString()).style = "padding: 3px 12%; margin: 0px 0.7%; border: 1px solid grey";
                document.getElementById("tab-btn-" + i.toString()).className = "btn btn-light";
                
                if (document.getElementById("tab-content-" + i.toString()) != null){
                    document.getElementById("tab-content-" + i.toString()).style = "display: none";
                }
            }
        }
        tab = tab_to_switch;
    }
}
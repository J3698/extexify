
addExtexifyButton()
addExtexifyPane()
addToggleExtexifyCallbacks()
hideExtexify()
clearCanvas()
addDrawingCallbacks()
addClassifyRequestInterval()
addReTypesetHandler()

function addExtexifyButton() {
    document.getElementsByClassName('formatting-buttons-wrapper')[0].innerHTML = `
      <span class="toggle-switch toggle-extexify-button" style="margin-left: 15px;">
        extexify
      </span>
    `;
}


function addExtexifyPane() {
    const editor = document.getElementById("editor")

    const extexifyPane = document.createElement("div");
    extexifyPane.classList.add("extexify-pane")
    editor.appendChild(extexifyPane)

    const backdrop = document.createElement("div");
    backdrop.classList.add("extexify-backdrop")
    editor.appendChild(backdrop)

    extexifyPane.innerHTML = `
        <div style = "padding: 11px; padding-top: 5px;">Draw a symbol below</div>

        <div>
            <canvas id = "extexify-canvas"></canvas>
        </div>

        <div style="font-size: 80%;">Click to Copy</div>


        <div class="predictions">
            <span class="prediction-wrapper">
                <span class="prediction">
                    <span class="totex"> </span>
                    <span class="actual"></span>
                </span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">
                    <span class="totex"> </span>
                    <span class="actual"></span>
                </span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">
                    <span class="totex"> </span>
                    <span class="actual"></span>
                </span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">
                    <span class="totex"> </span>
                    <span class="actual"></span>
                </span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">
                    <span class="totex"> </span>
                    <span class="actual"></span>
                </span>
            </span>
        <div>
    `

    const preds = document.getElementsByClassName("prediction")
    for (let i = 0; i < 5; i++) {
        let pred = preds[i];
        pred.onclick = function() {
            let pred2 = document.getElementsByClassName("prediction")[i]
            console.log(pred2);
            let actual = pred2.getElementsByClassName("actual")[0];
            console.log(actual);
            copySymbol(actual);
            toggleExtexify();
        }
    }
}


function copySymbol(pred) {
    var copyText = pred;
    var textArea = document.createElement("textarea");
    textArea.value = copyText.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("Copy");
    textArea.remove();
}

function hideExtexify() {
    document.getElementsByClassName("extexify-pane")[0].classList.add("fade-out")
    document.getElementsByClassName("extexify-backdrop")[0].classList.add("fade-out")
}


function addToggleExtexifyCallbacks() {
    document.getElementsByClassName("toggle-extexify-button")[0].onclick = toggleExtexify;
    document.getElementsByClassName("extexify-backdrop")[0].onclick = toggleExtexify;
}

function toggleExtexify() {
    console.log("toggling")
    document.getElementsByClassName("extexify-pane")[0].classList.toggle("fade-out")
    document.getElementsByClassName("extexify-backdrop")[0].classList.toggle("fade-out")

    var canvas = document.getElementById("extexify-canvas");
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    points = [[]];
}

function clearCanvas() {
    var canvas = document.getElementById("extexify-canvas");
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f6f6f6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

let points = [[]];
var shouldUpdate = true;
function addDrawingCallbacks() {
    var canvas = document.getElementById("extexify-canvas");
    var ctx = canvas.getContext("2d");

    var pos = {x: 0, y: 0};

    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mousedown', setPosition);
    canvas.addEventListener('mouseenter', setPosition);

    function setPosition(e) {
      var rect = canvas.getBoundingClientRect();
      pos.x = e.clientX - rect.left;
      pos.y = e.clientY - rect.top;
    }

    function draw(e) {
      if (e.buttons !== 1)  {
          if (points[points.length - 1].length !== 0) {
              points.push([]);
              shouldUpdate = true;
          }
          return;
      }
      //shouldUpdate = true;

      ctx.beginPath();

      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'black';

      ctx.moveTo(pos.x, pos.y);
      setPosition(e);
      ctx.lineTo(pos.x, pos.y);

      points[points.length - 1].push([pos.x, pos.y]);

      ctx.stroke();
    }
}


function addClassifyRequestInterval() {
    setInterval(function() {
        if (!shouldUpdate) {
            return
        }
        shouldUpdate = false;

        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (xhttp.readyState == 4 && xhttp.status == 200) {
                let jsonResponse = JSON.parse(xhttp.responseText);
                let top5 = jsonResponse['top5'];
                updatePredictionsHTML(top5);
            }
        };
        data = encodeURIComponent(JSON.stringify({"data": points}))
        xhttp.open("GET", "http://localhost/classify?points=" + data, true);
        xhttp.send(null);
    }, 150);
}


function addReTypesetHandler() {
    var script = document.createElement("script");
    script.innerHTML = `
        last = "1";
        setInterval(function() {
            var tag = document.getElementsByClassName("invisible")[0];
            if (tag.innerHTML != last) {
                MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
                last = tag.innerHTML;
            }
        }, 50);
    `
    document.body.appendChild(script);

    var updateTag = document.createElement("span");
    updateTag.classList.add("invisible");
    document.body.appendChild(updateTag);
}


function updatePredictionsHTML(topPredictions) {
    let [top1, top2, top3, top4, top5] = topPredictions;
    let [pred1, pred2, pred3, pred4, pred5] = document.getElementsByClassName("totex")
    pred1.innerHTML = '\\(' + top1 + '\\)';
    pred2.innerHTML = '\\(' + top2 + '\\)';
    pred3.innerHTML = '\\(' + top3 + '\\)';
    pred4.innerHTML = '\\(' + top4 + '\\)';
    pred5.innerHTML = '\\(' + top5 + '\\)';

    let [code1, code2, code3, code4, code5] = document.getElementsByClassName("actual")
    code1.innerHTML = top1;
    code2.innerHTML = top2;
    code3.innerHTML = top3;
    code4.innerHTML = top4;
    code5.innerHTML = top5;

    console.log()
    tag = document.getElementsByClassName("invisible")[0];
    if (tag.innerHTML != "0") {
        tag.innerHTML = "0";
    } else {
        tag.innerHTML = "1";
    }
}



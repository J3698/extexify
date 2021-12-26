
addExtexifyButton()
addExtexifyPane()
addToggleExtexifyCallbacks()
hideExtexify()
clearCanvas()
addDrawingCallbacks()
addClassifyRequestInterval()
addReTypesetHandler()
addThemeChangeHandler()

function addExtexifyButton() {
    var extexifyButton = document.createElement("span");
    extexifyButton.classList.add("toggle-switch");
    extexifyButton.classList.add("toggle-extexify-button");
    extexifyButton.style.marginLeft = "15px";
    extexifyButton.style.width = "fit-content";
    extexifyButton.innerHTML = "extexify";

    document.getElementsByClassName('formatting-buttons-wrapper')[0].appendChild(extexifyButton);
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
            let actual = pred2.getElementsByClassName("actual")[0];
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
    document.getElementsByClassName("extexify-pane")[0].classList.toggle("fade-out")
    document.getElementsByClassName("extexify-backdrop")[0].classList.toggle("fade-out")

    var canvas = document.getElementById("extexify-canvas");
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    points = [[]];
    updatePredictionsHTML([' ', ' ', ' ', ' ', ' ']);
}

function clearCanvas() {
    var canvas = document.getElementById("extexify-canvas");
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f6f6f6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

let points = [[]];
var shouldUpdate = false;
function addDrawingCallbacks() {
    var canvas = document.getElementById("extexify-canvas");
    var ctx = canvas.getContext("2d");

    var pos = {x: 0, y: 0};

    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', finish);
    canvas.addEventListener('mousedown', setPosition);
    canvas.addEventListener('mouseenter', setPosition);

    function setPosition(e) {
      var rect = canvas.getBoundingClientRect();
      pos.x = e.clientX - rect.left;
      pos.y = e.clientY - rect.top;
    }

    function finish(e) {
      if (points[points.length - 1].length !== 0) {
          points.push([]);
          shouldUpdate = true;
      }
    }

    function draw(e) {
      if (e.buttons !== 1)  {
          if (points[points.length - 1].length !== 0) {
              points.push([]);
              shouldUpdate = true;
          }
          return;
      }

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

        xhttp.open("POST", "https://extexify2.herokuapp.com/classify", true);
        // xhttp.open("POST", "http://127.0.0.1:8000/classify", true);
        xhttp.setRequestHeader('Content-Type', 'application/json');
        let data = JSON.stringify({"data": points});
        xhttp.send(data);
    }, 50);
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

function addThemeChangeHandler() {
    var script = document.createElement("script");
    script.innerHTML = `
    let etoolbar = document.getElementsByClassName("toolbar-header")[0];
    // .toolbar-editor

    let extexifyPane = document.getElementsByClassName('extexify-pane')[0];
    //color: white (or gray)

    let ecanvas = document.getElementById('extexify-canvas');
    let esidebar = document.getElementsByClassName("editor-sidebar")[0];

    let epreds = document.getElementsByClassName("prediction");

    setInterval(function() {
        let etoolbarStyles = window.getComputedStyle(etoolbar, null);
        let ebackgroundColor = etoolbarStyles['background-color'];
        extexifyPane.style.backgroundColor = ebackgroundColor;

        let esidebarStyles = window.getComputedStyle(esidebar, null);
        let ecanvasColor = esidebarStyles['background-color'];
        ecanvas.style.backgroundColor = ecanvasColor;

        let [m1, m2, m3] = ebackgroundColor.split("(")[1].split(")")[0].split(",").map(x=>+x);
        for (const epred of epreds) {
            if ((m1 + m2 + m3) / 3 > 127) {
                epred.style.color = 'rgb(0, 0, 0)';
            } else {
                epred.style.color = 'rgb(255, 255, 255)';
            }
        }

    }, 100);
    `

    document.body.appendChild(script);
}


function updatePredictionsHTML(topPredictions) {
    let [top1, top2, top3, top4, top5] = topPredictions;
    let [pred1, pred2, pred3, pred4, pred5] = document.getElementsByClassName("totex")

    let typeset = false;
    for (let i = 0; i < 5; i++) {
        if (topPredictions[i] !== " ") {
            typeset = true;
        }
    }

    if (typeset) {
        pred1.innerHTML = '\\(' + top1 + '\\)';
        pred2.innerHTML = '\\(' + top2 + '\\)';
        pred3.innerHTML = '\\(' + top3 + '\\)';
        pred4.innerHTML = '\\(' + top4 + '\\)';
        pred5.innerHTML = '\\(' + top5 + '\\)';
    } else {
        pred1.innerHTML = top1;
        pred2.innerHTML = top2;
        pred3.innerHTML = top3;
        pred4.innerHTML = top4;
        pred5.innerHTML = top5;
    }


    let [code1, code2, code3, code4, code5] = document.getElementsByClassName("actual")
    code1.innerHTML = top1;
    code2.innerHTML = top2;
    code3.innerHTML = top3;
    code4.innerHTML = top4;
    code5.innerHTML = top5;

    tag = document.getElementsByClassName("invisible")[0];
    if (tag.innerHTML != "0") {
        tag.innerHTML = "0";
    } else {
        tag.innerHTML = "1";
    }
}



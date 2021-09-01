addExtexifyButton()
addExtexifyPane()
addToggleExtexifyCallbacks()
hideExtexify()
clearCanvas()
addDrawingCallbacks()
addClassifyRequestInterval()

function addExtexifyButton() {
    document.getElementsByClassName('formatting-buttons-wrapper')[0].innerHTML = `
        <div class="formatting-buttons-wrapper">&nbsp;<!-- ngRepeat: button in shownButtons --><!-- ngIf: showMore -->
          <span class="toggle-switch toggle-detexify-button" style="margin-left: 15px;">
            detexify
          </span>
        </div>`;
}


function addExtexifyPane() {
    const editor = document.getElementById("editor")

    const detexifyPane = document.createElement("div");
    detexifyPane.classList.add("detexify-pane")
    editor.appendChild(detexifyPane)

    const backdrop = document.createElement("div");
    backdrop.classList.add("detexify-backdrop")
    editor.appendChild(backdrop)

    detexifyPane.innerHTML = `
        <div style = "padding: 11px; padding-top: 5px;">Draw a symbol below</div>

        <div>
            <canvas id = "detexify-canvas"></canvas>
        </div>

        <div class="predictions">
            <span class="prediction-wrapper">
                <span class="prediction">A</span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">Å</span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">ℜ</span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">B</span>
            </span>
            <span class="prediction-wrapper">
                <span class="prediction">C</span>
            </span>
        <div>
    `
}


function hideExtexify() {
    document.getElementsByClassName("detexify-pane")[0].classList.add("fade-out")
    document.getElementsByClassName("detexify-backdrop")[0].classList.add("fade-out")
}


function addToggleExtexifyCallbacks() {
    document.getElementsByClassName("toggle-detexify-button")[0].onclick = function() {
        console.log("toggling")
        document.getElementsByClassName("detexify-pane")[0].classList.toggle("fade-out")
        document.getElementsByClassName("detexify-backdrop")[0].classList.toggle("fade-out")

        var canvas = document.getElementById("detexify-canvas");
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        points = [[]];
    };

    document.getElementsByClassName("detexify-backdrop")[0].onclick = function() {
        console.log("toggling")
        document.getElementsByClassName("detexify-pane")[0].classList.toggle("fade-out")
        document.getElementsByClassName("detexify-backdrop")[0].classList.toggle("fade-out")

        var canvas = document.getElementById("detexify-canvas");
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        points = [[]];
    };

}


function clearCanvas() {
    var canvas = document.getElementById("detexify-canvas");
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f6f6f6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

let points = [[]];
function addDrawingCallbacks() {
    var canvas = document.getElementById("detexify-canvas");
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
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (xhttp.readyState == 4 && xhttp.status == 200) {
                let jsonResponse = JSON.parse(xhttp.responseText);
                let [top1, top2, top3] = jsonResponse['top3'];
                let [pred1, pred2, pred3, pred4, pred5] = document.getElementsByClassName("prediction")
                pred1.innerHTML = top1;
                pred2.innerHTML = top2;
                pred3.innerHTML = top3;
            }
        };
        data = encodeURIComponent(JSON.stringify({"data": points}))
        xhttp.open("GET", "http://localhost/classify?points=" + data, true);
        xhttp.send(null);
    }, 1000);
}




class BeforeAfter {
  constructor(enteryObject) {

      const beforeAfterContainer = document.querySelector(enteryObject.id);
      const before = beforeAfterContainer.querySelector('.bal-before');
      const beforeText = beforeAfterContainer.querySelector('.bal-beforePosition');
      const afterText = beforeAfterContainer.querySelector('.bal-afterPosition');
      const handle = beforeAfterContainer.querySelector('.bal-handle');
      var widthChange = 0;

      beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
      window.onresize = function () {
          beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
      }
      before.setAttribute('style', "width: 50%;");
      handle.setAttribute('style', "left: 50%;");

      //touch screen event listener
      beforeAfterContainer.addEventListener("touchstart", (e) => {

          beforeAfterContainer.addEventListener("touchmove", (e2) => {
              let containerWidth = beforeAfterContainer.offsetWidth;
              let currentPoint = e2.changedTouches[0].clientX;

              let startOfDiv = beforeAfterContainer.offsetLeft;

              let modifiedCurrentPoint = currentPoint - startOfDiv;

              if (modifiedCurrentPoint > 10 && modifiedCurrentPoint < beforeAfterContainer.offsetWidth - 10) {
                  let newWidth = modifiedCurrentPoint * 100 / containerWidth;

                  before.setAttribute('style', "width:" + newWidth + "%;");
                  afterText.setAttribute('style', "z-index: 1;");
                  handle.setAttribute('style', "left:" + newWidth + "%;");
              }
          });
      });

      //mouse move event listener
      beforeAfterContainer.addEventListener('mousemove', (e) => {
          let containerWidth = beforeAfterContainer.offsetWidth;
          widthChange = e.offsetX;
          let newWidth = widthChange * 100 / containerWidth;

          if (e.offsetX > 10 && e.offsetX < beforeAfterContainer.offsetWidth - 10) {
              before.setAttribute('style', "width:" + newWidth + "%;");
              afterText.setAttribute('style', "z-index:" + "1;");
              handle.setAttribute('style', "left:" + newWidth + "%;");
          }
      })

  }
}

function initComparisons() {
  var x, i;
  x = document.getElementsByClassName("img-comp-overlay");
  for (i = 0; i < x.length; i++) {
      compareImages(x[i]);
  }
  function compareImages(img) {
      var slider, img, clicked = 0, w, h;
      w = img.offsetWidth;
      h = img.offsetHeight;
      img.style.width = (w / 2) + "px";
      slider = document.createElement("DIV");
      slider.setAttribute("class", "img-comp-slider");
      img.parentElement.insertBefore(slider, img);
      slider.style.top = (h / 2) - (slider.offsetHeight / 2) + "px";
      slider.style.left = (w / 2) - (slider.offsetWidth / 2) + "px";
      slider.addEventListener("mousedown", slideReady);
      window.addEventListener("mouseup", slideFinish);
      slider.addEventListener("touchstart", slideReady);
      window.addEventListener("touchend", slideFinish);

      function slideReady(e) {
          e.preventDefault();
          clicked = 1;
          window.addEventListener("mousemove", slideMove);
          window.addEventListener("touchmove", slideMove);
      }

      function slideFinish() {
          clicked = 0;
      }

      function slideMove(e) {
          var pos;
          if (clicked == 0) return false;
          pos = getCursorPos(e);
          if (pos < 0) pos = 0;
          if (pos > w) pos = w;
          slide(pos);
      }

      function getCursorPos(e) {
          var a, x = 0;
          e = (e.changedTouches) ? e.changedTouches[0] : e;
          a = img.getBoundingClientRect();
          x = e.pageX - a.left;
          x = x - window.pageXOffset;
          return x;
      }

      function slide(x) {
          img.style.width = x + "px";
          slider.style.left = img.offsetWidth - (slider.offsetWidth / 2) + "px";
      }
  }
}

document.addEventListener('DOMContentLoaded', (event) => {
  initComparisons();
});
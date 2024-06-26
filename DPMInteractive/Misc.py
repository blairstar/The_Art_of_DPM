 

js_head = "<script>" + open("ExtraBlock.js").read() + "</script>" \
          # + """ <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/katex@0.15.3/dist/katex.min.js" integrity="sha384-0fdwu/T/EQMsQlrHCCHoH10pkPLlKA1jL5dFyUOvB3lfeT2540/2g6YgSi2BL14p" crossorigin="anonymous"></script> """


js_load = """
        function load_callback() {
            insert_markdown();
            add_switch();
        }
          """


g_css = """
        .bgc {background-color: #ffffff; border-width: 0 !important}
        .first_demo span{font-size: 140%; font-weight: bold; color: blue}
        .first_md span{font-size: 140%; font-weight: bold; color: orange}
        .normal span{font-size: 100%; font-weight: normal; color: black}
        .second span{font-size: 100%; font-weight: bold; color: blue}
        .mds div{margin-top: 10px; margin-bottom: 20px; margin-left:10px; margin-right:10px; font-size:16px;}
        .gps div{margin-top: 10px; margin-bottom: 20px;}

        .switchbar {position: relative; display: inline-block; width: 60px; height: 30px; margin-left: 10px; margin-right: 10px}
        .switchbar input {opacity: 0; width: 0; height: 0;}

        .switchslider {position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
                       border-radius: 30px; background-color: orange; -webkit-transition: .4s; transition: .4s;}

        .switchslider:before {position: absolute; content: ""; height: 22px; width: 22px; border-radius: 50%;
                              left: 4px; bottom: 4px; background-color: white; -webkit-transition: .4s;
                              transition: .4s;}

        input:checked + .switchslider {background-color: orange;}

        input:checked + .switchslider:before {-webkit-transform: translateX(26px); -ms-transform: translateX(26px);
                                              transform: translateX(26px);}
                                              
        ul,ol { display: block; list-style-type: disc;
                padding-inline-start: 40px;}
        ul,ol:last-of-type { list-style-position: outside;}
        ol { list-style-type: decimal ;}
        li { display: list-item;}
        """

g_latex_del = [
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": "\\(", "right": "\\)", "display": False},
                {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                {"left": "\\begin{aligned}", "right": "\\end{aligned}", "display": True}
              ]

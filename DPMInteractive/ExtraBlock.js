


async function write_markdown() {
    let names = ["introduction", "transform", "likelihood", "posterior", "forward_process", "backward_process",
                 "fit_posterior", "posterior_transform", "deconvolution", "reference", "about"];
    // names = names.slice(-1)
    
    let data = await fetch("file/data.json").then(response => response.json());
    
    names.forEach((name, index) => {
        let elem_zh = document.getElementById("md_" + name + "_zh");
        if (elem_zh != null) { data[name+"_zh"] = elem_zh.outerHTML; }

        const elem_en = document.getElementById("md_" + name + "_en");
        if (elem_en != null) { data[name+"_en"] = elem_en.outerHTML; }
    });
    
    let a = document.createElement('a');
    a.href = "data:application/octet-stream," + encodeURIComponent(JSON.stringify(data));
    a.download = "data.json";
    a.click();
}


async function insert_markdown() {
    let names = ["introduction", "transform", "likelihood", "posterior", "forward_process", "backward_process", 
                 "fit_posterior", "posterior_transform", "deconvolution", "reference", "about"];

    let data = await fetch("file/data.json").then(response => response.json());
    
    for (let i = 0; i < names.length; i++) {
        name = names[i];
        const markdown_zh = document.createElement('div');
        markdown_zh.id = "md_" + name + "_zh";
        markdown_zh.style.display = "none";
        
        markdown_zh.innerHTML = data[name+"_zh"] 
        // fetch('file/Markdown/zh/' + name + ".html").then(response => response.text()).then(text => markdown_zh.innerHTML = text)

        const markdown_en = document.createElement('div');
        markdown_en.id = "md_" + name + "_en";
        markdown_en.style.display = "block";
        markdown_en.innerHTML = data[name+"_en"] 
         
        // fetch('file/Markdown/en/' + name + ".html").then(response => response.text()).then(text => markdown_en.innerHTML = text)

        const accord = document.getElementById(name).children[2];
        accord.appendChild(markdown_zh);
        accord.appendChild(markdown_en);
    }
}


function control_language() {
    const names = ["introduction", "transform", "likelihood", "posterior", "forward_process",
        "backward_process", "fit_posterior", "posterior_transform", "deconvolution", "reference", "about"];

    var is_zh = document.getElementById("switch_language").checked;
    for (let i = 0; i < names.length; i++) {
        name = names[i];

        zh_display = is_zh ? "block" : "none";
        en_display = is_zh ? "none" : "block";

        md_zh = document.getElementById("md_" + name + "_zh");
        md_en = document.getElementById("md_" + name + "_en");
        md_zh.style.display = zh_display;
        md_en.style.display = en_display;
    }
}

function control_accordion() {
    open = document.getElementById("switch_accordion").checked;
    all_accords = document.getElementsByClassName("label-wrap");

    for (let i = 0; i < all_accords.length; i++) {
        const accord = all_accords[i];
        const is_open = accord.classList.contains("open");
        if (is_open != open) {
            accord.click();
        }
    }
}


function create_switch(id, left_text, right_text, callback) {
    var input = document.createElement("input");
    input.type = "checkbox";
    input.id = id;
    input.checked = false;
    input.addEventListener("change", callback);

    var slider = document.createElement("span");
    slider.className = "switchslider";

    var switchbar = document.createElement("label");
    switchbar.className = "switchbar";

    switchbar.appendChild(input);
    switchbar.appendChild(slider);

    var div = document.createElement("div");
    div.style.lineHeight = "30px";
    div.style.fontSize = "18px";

    var left_label = document.createElement("span");
    left_label.innerText = left_text;
    var right_label = document.createElement("span");
    right_label.innerText = right_text;

    div.appendChild(left_label);
    div.appendChild(switchbar);
    div.appendChild(right_label);

    return div;
}


function add_switch() {
    switch_accordion = create_switch("switch_accordion", "Collapse", "Expand", control_accordion);
    switch_language = create_switch("switch_language", "English", "Chinese", control_language);

    switch_accordion.style.float = "right";
    switch_language.style.float = "left";

    var state = document.createElement("span");
    state.appendChild(switch_language);
    state.appendChild(switch_accordion);
    state.style.marginBottom = "20px"

    const elem = document.getElementById("introduction");
    elem.insertAdjacentElement("beforebegin", state);

    switch_accordion.children[1].click();
    switch_accordion.children[1].click();
}


function katex_render(name) {
    const elem = document.getElementById(name);
    if (elem == null) { return; }
    var text = elem.innerText.replaceAll("{underline}", "_");
    text = "\\begin{align}\n" + text + "\n\\end{align}";
    katex.render(text, elem, {displayMode: true});
}


function insert_special_formula() {
    katex_render("zh_fit_0");
    katex_render("zh_fit_1");
    katex_render("zh_fit_2");
    katex_render("en_fit_0");
    katex_render("en_fit_1");
    katex_render("en_fit_2");
}

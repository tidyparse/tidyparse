/* Parser.js
 * written by Colin Kuebler 2012
 * Part of LDT, dual licensed under GPLv3 and MIT
 * Generates a tokenizer from regular expressions for TextareaDecorator
 */

function Parser(rules, i) {
    /* INIT */
    const api = this;

    // variables used internally
    var i = i ? 'i' : '';
    let parseRE = null;
    const ruleSrc = [];
    const ruleMap = {};

    api.add = function (rules) {
        for (const rule in rules) {
            const s = rules[rule].source;
            ruleSrc.push(s);
            ruleMap[rule] = new RegExp('^(' + s + ')$', i);
        }
        parseRE = new RegExp(ruleSrc.join('|'), 'g' + i);
    };
    api.tokenize = function (input) {
        return input.match(parseRE);
    };
    api.identify = function (token) {
        for (const rule in ruleMap) {
            if (ruleMap[rule].test(token)) {
                return rule;
            }
        }
    };

    api.add(rules);

    return api;
}

/* TextareaDecorator.js
 * written by Colin Kuebler 2012
 * Part of LDT, dual licensed under GPLv3 and MIT
 * Builds and maintains a styled output layer under a textarea input layer
 */

function TextareaDecorator(textarea, parser) {
    /* INIT */
    const api = this;

    // construct editor DOM
    const parent = document.createElement("div");
    const output = document.createElement("pre");
    parent.appendChild(output);
    const label = document.createElement("label");
    parent.appendChild(label);
    // replace the textarea with RTA DOM and reattach on label
    textarea.parentNode.replaceChild(parent, textarea);
    label.appendChild(textarea);
    // transfer the CSS styles to our editor
    parent.className = 'ldt ' + textarea.className;
    textarea.className = '';
    // turn off built-in spellchecking in firefox
    textarea.spellcheck = false;
    // turn off word wrap
    textarea.wrap = "off";

    // coloring algorithm
    const color = function (input, output, parser) {
        const oldTokens = output.childNodes;
        const newTokens = parser.tokenize(input);
        let firstDiff, lastDiffNew, lastDiffOld;
        // find the first difference
        for (firstDiff = 0; firstDiff < newTokens.length && firstDiff < oldTokens.length; firstDiff++)
            if (newTokens[firstDiff] !== oldTokens[firstDiff].textContent) break;
        // trim the length of output nodes to the size of the input
        while (newTokens.length < oldTokens.length)
            output.removeChild(oldTokens[firstDiff]);
        // find the last difference
        for (lastDiffNew = newTokens.length - 1, lastDiffOld = oldTokens.length - 1; firstDiff < lastDiffOld; lastDiffNew--, lastDiffOld--)
            if (newTokens[lastDiffNew] !== oldTokens[lastDiffOld].textContent) break;
        // update modified spans
        for (; firstDiff <= lastDiffOld; firstDiff++) {
            oldTokens[firstDiff].className = parser.identify(newTokens[firstDiff]);
            oldTokens[firstDiff].textContent = oldTokens[firstDiff].innerText = newTokens[firstDiff];
        }
        // add in modified spans
        for (let insertionPt = oldTokens[firstDiff] || null; firstDiff <= lastDiffNew; firstDiff++) {
            const span = document.createElement("span");
            span.className = parser.identify(newTokens[firstDiff]);
            span.textContent = span.innerText = newTokens[firstDiff];
            output.insertBefore(span, insertionPt);
        }
    };

    api.input = textarea;
    api.output = output;
    api.update = function () {
        const input = textarea.value;
        if (input) {
            color(input, output, parser);
            // determine the best size for the textarea
            const lines = input.split('\n');
            // find the number of columns
            let maxlen = 0, curlen;
            for (let i = 0; i < lines.length; i++) {
                // calculate the width of each tab
                let tabLength = 0, offset = -1;
                while ((offset = lines[i].indexOf('\t', offset + 1)) > -1) {
                    tabLength += 7 - (tabLength + offset) % 8;
                }
                let curlen = lines[i].length + tabLength;
                // store the greatest line length thus far
                maxlen = maxlen > curlen ? maxlen : curlen;
            }
            textarea.cols = maxlen + 1;
            textarea.rows = lines.length + 2;
        } else {
            // clear the display
            output.innerHTML = '';
            // reset textarea rows/cols
            textarea.cols = textarea.rows = 1;
        }
    };

    // detect all changes to the textarea,
    // including keyboard input, cut/copy/paste, drag & drop, etc
    if (textarea.addEventListener) {
        // standards browsers: oninput event
        textarea.addEventListener("input", api.update, false);
    } else {
        // MSIE: detect changes to the 'value' property
        textarea.attachEvent("onpropertychange",
            function (e) {
                if (e.propertyName.toLowerCase() === 'value') {
                    api.update();
                }
            }
        );
    }
    // initial highlighting
    api.update();

    return api;
}
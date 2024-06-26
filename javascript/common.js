const LIGHT_THEME = "light";
const DARK_THEME = "dark";
const SYSTEM_THEME = "system";

/**
 * Changes the theme in the DOM.
 * @param theme The selected theme.
 */
function ChangeDisplayTheme(theme)
{
    document.documentElement.dataset.appliedMode = GetValidThemeForIdentifier(theme);
}

/**
 * Gets the user's preferred theme from the local storage.
 * @returns {string|string} The preferred theme, or "system" if none.
 */
function GetThemePreference()
{
    return localStorage.getItem("theme") || "system";
}

/**
 * Sets the user's preferred theme to local storage.
 * @param theme The selected theme.
 */
function SetThemePreference(theme)
{
    localStorage.setItem("theme", theme);

    ChangeDisplayTheme(theme);
}

/**
 * Returns the theme to use based on the identifier.
 * @param {string} theme The theme identifier.
 * @returns {string} The theme to use.
 */
function GetValidThemeForIdentifier(theme)
{
    switch (theme)
    {
        case SYSTEM_THEME:
            let prefersLight = matchMedia("(prefers-color-scheme: light)").matches;
            if (prefersLight)
                return LIGHT_THEME;

            let prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;
            if (prefersDark)
                return DARK_THEME;
            break;
        case LIGHT_THEME:
        case DARK_THEME:
            return theme;
        default:
            return DARK_THEME;
    }
}

/**
 * Handles theme selection.
 */
function OnThemeChanged()
{
    let dropdown = document.getElementById("themeDropdown");
    let value = dropdown.value;

    switch (value)
    {
        case LIGHT_THEME:
        case DARK_THEME:
        case SYSTEM_THEME:
            SetThemePreference(value);
            break;
        default:
            SetThemePreference(SYSTEM_THEME);
            break;
    }
}

/**
 * Loads the site theme.
 */
function LoadTheme()
{
    let preference = GetThemePreference();

    ChangeDisplayTheme(preference);
}

/**
 * Loads external HTML elements dynamically.
 */
async function LoadElements()
{
    let includeElements = document.querySelectorAll('[data-include]');

    for (const element of includeElements)
    {
        let name = element.getAttribute("data-include");
        let file = '/elements/' + name + '.html';

        let response = await fetch(file);
        let data = await response.text();

        ReplaceElement(element, data);
    }
}

/**
 * Replaces an element with another, while executing scripts.
 * Will not work if the replacement element has more than 2 root nodes.
 * @param element {Element} The target element.
 * @param html {string} The HTML to replace the element with.
 */
function ReplaceElement(element, html)
{
    let next = element.nextElementSibling;
    element.outerHTML = html;

    let newElement = next.previousElementSibling;

    for (const originalScript of newElement.getElementsByTagName("script"))
    {
        const newScriptEl = document.createElement("script");

        for (const attr of originalScript.attributes)
        {
            newScriptEl.setAttribute(attr.name, attr.value)
        }

        const scriptText = document.createTextNode(originalScript.innerHTML);
        newScriptEl.appendChild(scriptText);

        originalScript.parentNode.replaceChild(newScriptEl, originalScript);
    }
}

/**
 * Opens the page sidebar.
 */
function OpenSidebar()
{
    let sideBar = document.getElementById("sideBar");
    let isShown = sideBar.classList.contains("shown");

    sideBar.classList.toggle("shown", !isShown);
}

/**
 * Hides the sidebar if clicked outside of its bounds.
 * Only for mobile.
 * @param {MouseEvent} e The mouse click event.
 */
function HandleClick(e)
{
    let sideBar = document.getElementById("sideBar");
    let sideBarButton = document.getElementById("sideBarButton");

    if (!sideBar.contains(e.target) && !sideBarButton.contains(e.target))
    {
        sideBar.classList.toggle("shown", false);
    }
}

const POSTS_PATH = "/resources/data/posts.json";
const KNOWLEDGE_PATH = "/resources/data/knowledge.json";

/**
 * Retrieves blog post data from a JSON file.
 * @param {boolean} isPosts If true, fetch content from the posts file, otherwise, knowledge.
 * @param {...receivePostsCallback} callbackFunctions The callbacks that will receive the data.
 */
async function RetrieveBlogData(isPosts, ...callbackFunctions)
{
    let response = await fetch(isPosts ? POSTS_PATH : KNOWLEDGE_PATH);
    let json = await response.json();

    callbackFunctions.forEach(x => x.apply(null, new Array(json)));
}

/**
 * Generates a post or article preview item.
 * @param {object} articleData The JSON data.
 * @returns {HTMLDivElement} The new article or post preview.
 */
function GenerateArticle(articleData)
{
    let createdArticle = document.createElement("div");
    createdArticle.classList.add("listItem", "listItemVertical");

    let editedString = !articleData.edited ? "Never Edited" : `Edited on <b>${articleData.edited}</b>`;

    createdArticle.innerHTML = `<a href="${articleData.url}"><h3 class="postHeader">${articleData.title}</h3></a>`;
    createdArticle.innerHTML += `<p class="postInformation postHeaderInformation">Posted on <b>${articleData.date}</b><span class="postInformationBullet"></span>${editedString}</p>`;
    createdArticle.innerHTML += `<p class="postDescription">${articleData.description}</p>`;

    return createdArticle;
}

/**
 * Callback for receiving post data.
 * @callback receivePostsCallback
 * @param {object} data The JSON data.
 */

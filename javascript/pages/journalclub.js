LoadTheme();

window.onload = PageStartup;

async function PageStartup()
{
	await LoadElements();
	await RetrieveBlogData(false, LoadArticles);

	window.addEventListener("mouseup", (e) => HandleClick(e));
}

function LoadArticles(fileContent)
{
	let paperList = document.getElementById("paperList");

	for (const value of fileContent)
	{
		let createdArticle = GenerateArticle(value);

		paperList.appendChild(createdArticle);
	}
}

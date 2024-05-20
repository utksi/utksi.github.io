---
layout: default
title: "Journal Club - UT's Blog"
---

# Journal Club

Here I write summaries of papers I find interesting.

<div class="listContainer">
  {% for article in site.knowledge %}
  <div class="listItem">
    <a href="{{ article.url }}">{{ article.title }}</a> - {{ article.date | date_to_string }}
    <p>{{ article.description }}</p>
  </div>
  <div class="listSeparator"></div>
  {% endfor %}
</div>


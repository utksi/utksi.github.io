---
layout: default
title: "Knowledge - Utkarsh"
---

# Knowledge Base

**Still porting. Everything will eventually be here.**

Here is my knowledge base. I write stuff down here if information about it is relatively hard to find.

<div class="listContainer">
  {% for article in site.knowledge %}
  <div class="listItem">
    <a href="{{ article.url }}">{{ article.title }}</a> - {{ article.date | date_to_string }}
    <p>{{ article.description }}</p>
  </div>
  <div class="listSeparator"></div>
  {% endfor %}
</div>


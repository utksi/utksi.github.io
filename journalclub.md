---
layout: default
title: "Journal Club - Utkarsh"
---

# Journal Club

**Still porting. Everything will eventually be here.**

Here I write summaries of papers I find interesting.

<div class="listContainer">
  {% for paper in site.journalclub %}
  <div class="listItem">
    <a href="{{ paper.url }}">{{ paper.title }}</a> - {{ paper.date | date_to_string }}
    <p>{{ paper.description }}</p>
  </div>
  <div class="listSeparator"></div>
  {% endfor %}
</div>


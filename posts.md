---
layout: default
title: "Posts - UT's Blog"
---

# Blog Posts

Here is my personal blog! It's not very active, but I sometimes get enough motivation to write something.

<div class="listContainer">
  {% for post in site.posts %}
  <div class="listItem">
    <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date_to_string }}
    <p>{{ post.description }}</p>
  </div>
  <div class="listSeparator"></div>
  {% endfor %}
</div>


---
layout: home
title: "Home - UT's Blog"
---

# Home Page

Hi! Welcome to my personal website. I'm [Utkarsh](https://github.com/utksi),

    - **Primary OS**: macOS
    - **Secondary OS**: arch (yes, I know)
    - **Primary IDE**: VSCode 2024
    - **Secondary IDE**: neovim

## My Programming Languages

Here's a list of programming languages I work with or have worked with before!

<div class="listContainer">
  <div class="listItem">
    <img src="/resources/langs/csharp.png">
    <div class="languagesText">
      **C#**  
      It's my favorite language so far. I do most of my programs in it, I can't live without it.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/c.png">
    <div class="languagesText">
      **C**  
      Barely use it, since I almost never do low-level things, and if I do, I either use C++ or use P/Invoke on C#.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/cpp.png">
    <div class="languagesText">
      **C++**  
      The first language I ever tried to learn, didn't go well, obviously. Now I use it for low-level things.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/java.png">
    <div class="languagesText">
      **Java**  
      I tried to learn it before I settled with C#. I barely use it anymore.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/js.png">
    <div class="languagesText">
      **JavaScript**  
      I absolutely despise it, but it's a necessity for websites.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/html.png">
    <div class="languagesText">
      **HTML**  
      I also absolutely despise it, but it's at least less insane than JS.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/css.png">
    <div class="languagesText">
      **CSS**  
      This wretched thing should not be alive. It makes websites look pretty though.
    </div>
  </div>
</div>

## Latest Posts

<details>
  <summary>Show Latest Posts</summary>
  <ul>
    {% for post in site.posts limit:3 %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date_to_string }}
      <p>{{ post.description }}</p>
    </li>
    {% endfor %}
  </ul>
  <p><a href="/posts">See all posts</a></p>
</details>

## Latest Knowledge Articles

<details>
  <summary>Show Latest Knowledge Articles</summary>
  <ul>
    {% for article in site.knowledge limit:3 %}
    <li>
      <a href="{{ article.url }}">{{ article.title }}</a> - {{ article.date | date_to_string }}
      <p>{{ article.description }}</p>
    </li>
    {% endfor %}
  </ul>
  <p><a href="/knowledge">See all knowledge articles</a></p>
</details>


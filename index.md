---
layout: home
title: "Home - Utkarsh Singh"
---

# Home Page

**Still porting. Everything will eventually be here. This message will disappear when it happens.**

Hi! Welcome to my personal website. I'm [Utkarsh](https://github.com/utksi).

- **Primary OS**: macOS    
- **Secondary OS**: arch (yes, I know)
- **Primary IDE**: VSCode 2024
- **Secondary IDE**: neovim


<div>
    <h2>Research notes [-3:0]</h2>
      <ul>
      {% for article in site.knowledge limit:3 %}
      <li>
      <a href="{{ article.url }}">{{ article.title }}</a> - {{ article.date | date_to_string }}
      <p>{{ article.description }}</p>
      </li>
      {% endfor %}
      </ul>
    <h2>Journal Club [-3:0]</h2>
      <ul>
      {% for paper in site.journalclub limit:3 %}
      <li>
      <a href="{{ paper.url }}">{{ paper.title }}</a> - {{ paper.date | date_to_string }}
      <p>{{ paper.description }}</p>
      </li>
      {% endfor %}
      </ul>
    <h2>Personal [-3:0]</h2>
      <ul>
      {% for post in site.posts limit:3 %}
      <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date_to_string }}
      <p>{{ post.description }}</p>
      </li>
      {% endfor %}
      </ul>
</div>

## My Programming Languages

**Holds mostly true till a couple years ago. Now I have picked up fortran! huehuehue.**

Here's a list of programming languages I work with or have worked with before!

<div class="listContainer">
  <div class="listItem">
    <img src="/resources/langs/python.png">
    <div class="languagesText">
      <strong>Python</strong>  
      I do most of my work with this (sometimes as a wrapper for C).
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/fortran.png">
    <div class="languagesText">
      <strong>Fortran</strong>  
      Most solvers I work with are written in F2003 or F2008 standard.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/c.png">
    <div class="languagesText">
      <strong>C</strong>  
      No shade, but for my usecase it's not very useful (python/C++ wrapper works).
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/cpp.png">
    <div class="languagesText">
      <strong>C++</strong>  
      I use it for low-level things.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/java.png">
    <div class="languagesText">
      <strong>Java</strong>  
      死ぬ
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/js.png">
    <div class="languagesText">
      <strong>JavaScript</strong>  
      It's a necessity unfortunately.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/html.png">
    <div class="languagesText">
      <strong>HTML</strong>  
       Useful. Simple.
    </div>
  </div>
  <div class="listSeparator"></div>
  <div class="listItem">
    <img src="/resources/langs/css.png">
    <div class="languagesText">
      <strong>CSS</strong>  
      Useful - yes (website prettifier). Necessary - no (simple is best).
    </div>
  </div>
</div>

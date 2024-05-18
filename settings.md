---
layout: default
title: "Settings - UT's Blog"
---

<h1>Settings Page</h1>
<p>
  This page contains client-side settings to modify how the website is displayed.
  <br>
  Right now it only contains basic cosmetic settings such as the theme.
</p>
<fieldset>
  <legend>Color Mode</legend>
  Controls the color mode of the site, such as dark, light or system defined.
  <br>
  Dark mode is recommended if you value your corneas.
  <br><br>

  <label for="themeDropdown">Current Value:</label>
  <select name="themeDropdown" id="themeDropdown">
    <option value="light">Light</option>
    <option value="dark">Dark</option>
    <option value="system">System</option>
  </select>
</fieldset>

<script src="/javascript/common.js"></script>
<script src="/javascript/pages/settings.js"></script>


# Contributing to Atari

Looking to contribute something to Atari? Here's how you can help.

Please take a moment to review this document in order to make the contribution process easy and effective for everyone involved.

Following these guidelines helps to communicate that you respect the time of the developer managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue or assessing patches and features.

## Using the issue tracker

The [issue tracker](https://github.com/Kaixhin/Atari/issues) is the preferred channel for [bug reports](#bug-reports), [feature requests](#feature-requests) and [submitting pull requests](#pull-requests), but please respect the following restrictions:

- Please **do not** use the issue tracker for personal support requests. This includes asking for help on your own code.

- Please **do not** derail or troll issues. Keep the discussion on topic and respect the opinions of others.

- Please **do not** post comments consisting solely of "+1" or ":thumbsup:". Use [GitHub's "reactions" feature](https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments) instead. I reserve the right to delete comments which violate this rule.

- Please **do not** ask for this repository to be ported to another language/outside of Torch7.

## Bug reports

A bug is a _demonstrable problem_ that is caused by the code in the repository. Good bug reports are extremely helpful - thank you!

Guidelines for bug reports:

1. **Use the GitHub issue search** &mdash; check if the issue has already been reported.

2. **Check if the issue has been fixed** &mdash; try to reproduce it using the latest `master` or development branch in the repository.

3. **Isolate the problem** &mdash; ideally create test case that is within reason, preferably within 100 lines of code.

A good bug report shouldn't leave others needing to chase you up for more information. Please try to be as detailed as possible in your report. What is your environment? What steps will reproduce the issue? What OS do you experience the problem? What would you expect to be the outcome? All these details will help people to fix any potential bugs.

Bugs can be somewhat difficult to isolate in machine learning code, but most of the above should still be applicable.

## Feature requests

Feature requests are welcome to be filed. This project has one primary developer who works on this during his free time, so please keep that in mind.

Before opening a feature request, please take a moment to find out whether your idea fits with the scope and aims of the project. It's up to *you* to make a strong case to convince the project's developer of the merits of this feature. Please provide as much detail and context as possible.

## Pull requests

Good pull requests - patches, improvements, new features - are a fantastic help. They should remain focused in scope **and avoid containing unrelated commits.**

**Please ask first** before embarking on any significant pull request (e.g. implementing features, refactoring code), otherwise you risk spending a lot of time working on something that the project's developers might not want to merge into the project.

Please adhere to the [coding guidelines](#code-guidelines) used throughout the project (indentation, accurate comments, etc.) and any other requirements (such as test coverage).

Adhering to the following this process is the best way to get your work included in the project:

1. [Fork](https://help.github.com/articles/fork-a-repo) the project, clone your fork, and configure the remotes:

   ```bash
   # Clone your fork of the repo into the current directory
   git clone https://github.com/<your-username>/Atari.git
   # Navigate to the newly cloned directory
   cd Atari
   # Assign the original repo to a remote called "upstream"
   git remote add upstream https://github.com/Kaixhin/Atari.git
   ```

2. If you cloned a while ago, get the latest changes from upstream:

   ```bash
   git checkout master
   git pull upstream master
   ```

3. Create a new topic branch (off the main project development branch) to contain your feature, change, or fix:

   ```bash
   git checkout -b <topic-branch-name>
   ```

4. Commit your changes in logical chunks. Please try to adhere to these [git commit message guidelines](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html). Use Git's [interactive rebase](https://help.github.com/articles/about-git-rebase) feature to tidy up your commits before making them public.

5. Locally merge (or rebase) the upstream development branch into your topic branch:

   ```bash
   git pull [--rebase] upstream master
   ```

6. Push your topic branch up to your fork:

   ```bash
   git push origin <topic-branch-name>
   ```

7. [Open a Pull Request](https://help.github.com/articles/using-pull-requests/) with a clear title and description.

### Code guidelines

Please try to follow the general coding style in this repository. Loosely:

- 2 space indentation (spaces or tab)
- camelCase
- Comments encouraged (especially those including mathematical equations)

### License

By contributing your code, you agree to license your contribution under the [MIT License](https://github.com/Kaixhin/Atari/blob/master/LICENSE.md).

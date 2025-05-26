# epic

Thank you for your interest in our work.

Unfortunately, we are unable to share the original code at this time, as it was developed internally at Huawei and is subject to company policies that prohibit external distribution. We have discussed this with the company on multiple occasions, but unfortunately, the internal policies are often rigid and difficult to change.

That said, we fully understand the importance of open and reproducible research. To address this, we have created a new repository and plan to reimplement the code from scratch in my spare time. Of course, we still need to confirm with the company that the first author independently reimplementing the code outside the organization does not violate internal policies—as some rules may require using a different programming language to ensure compliance.

In the meantime, we are more than happy to share some implementation insights. Functionally, the core of EPIC is indeed as simple as described in the paper: it only requires recomputing the first few tokens of each chunk. If you're aiming for functional validation rather than performance optimization, applying an appropriate mask should suffice to achieve the intended behavior. We hope this helps with your implementation.

--------Update 2025.05.26---------

The good news is that we are currently implementing a PIC mechanism for a multimodal model (MPIC), and this version is developed entirely on our personal machines, independent of any corporate restrictions. We may eventually share the link to the MPIC repository (which is currently private) instead. MPIC follows the same core principles as EPIC; the only major difference is that MPIC is designed for multimodal settings.

/* =========================================
TINYHEALTH — JAVASCRIPT
========================================= */

/* =========================================
CONFIGURATION
========================================= */

const HUGGINGFACE_ENDPOINT =
    "https://samarthshukla-TinyHealth1-inference.hf.space/check_claim";

/* =========================================
DOM ELEMENTS
========================================= */

const claimInput =
    document.getElementById("claimInput");

const checkButton =
    document.getElementById("checkButton");

const statusMessage =
    document.getElementById("statusMessage");

const statusText =
    document.getElementById("statusText");

const spinner =
    document.getElementById("spinner");

const responseTextDiv =
    document.getElementById("responseText");

const initialMessage =
    document.getElementById("initialMessage");

const sourcesListDiv =
    document.getElementById("sourcesList");

/* =========================================
BUTTON STATE
========================================= */

function updateButtonState() {


    checkButton.disabled =
        !claimInput.value.trim();


}

/* Enable button when user types */

claimInput.addEventListener(
    "input",
    updateButtonState
);

/* =========================================
EXAMPLE CLAIMS
========================================= */

const exampleChips =
    document.querySelectorAll(".example-chip");

exampleChips.forEach(chip => {


    chip.addEventListener(
        "click",
        () => {

            /* Put example text into input */

            claimInput.value =
                chip.textContent.trim();


            /* Enable analyze button */

            updateButtonState();


            /* Focus input */

            claimInput.focus();

        }
    );


});

/* =========================================
ENTER KEY SUPPORT
========================================= */

claimInput.addEventListener(
    "keydown",
    event => {


        if (
            event.key === "Enter" &&
            !checkButton.disabled
        ) {

            handleFactCheck();

        }

    }


);

/* =========================================
ANALYZE BUTTON
========================================= */

checkButton.addEventListener(
    "click",
    handleFactCheck
);

/* =========================================
MAIN FACT CHECK FUNCTION
========================================= */

async function handleFactCheck() {


    /* Get user claim */

    const userClaim =
        claimInput.value.trim();


    /* Maximum API waiting time */

    const TIMEOUT_MS =
        60000;


    /* =====================================
       VALIDATION
    ===================================== */

    if (!userClaim) {

        setStatus(
            "Please enter a health claim first.",
            "error",
            false
        );

        return;

    }


    /* =====================================
       RESET PREVIOUS RESULT
    ===================================== */

    initialMessage.classList.add("hidden");

    responseTextDiv.classList.add("hidden");

    responseTextDiv.innerHTML = "";

    sourcesListDiv.innerHTML = "";


    /* =====================================
       LOADING STATE
    ===================================== */

    setStatus(
        "Analyzing your claim using TinyHealth AI...",
        "loading",
        true
    );


    checkButton.disabled = true;

    checkButton.innerHTML =
        "⏳ Analyzing...";


    try {


        /* =================================
           CREATE REQUEST CONTROLLER
        ================================= */

        const controller =
            new AbortController();


        const timeoutId =
            setTimeout(
                () => controller.abort(),
                TIMEOUT_MS
            );


        /* =================================
           CALL AI MODEL
        ================================= */

        const resultText =
            await callCustomModelAPI(
                userClaim,
                controller.signal
            );


        /* Stop timeout */

        clearTimeout(timeoutId);


        /* =================================
           DISPLAY RESPONSE
        ================================= */

        responseTextDiv.innerHTML =
            formatResponse(resultText);


        responseTextDiv.classList.remove(
            "hidden"
        );


        /* Model information */

        sourcesListDiv.innerHTML =
            "🤖 Analysis powered by TinyHealth-1 model";


        /* Success message */

        setStatus(
            "Analysis complete!",
            "success",
            false
        );


        /* Scroll result into view */

        setTimeout(() => {

            responseTextDiv.scrollIntoView({
                behavior: "smooth",
                block: "nearest"
            });

        }, 150);


    }


    /* =====================================
       ERROR HANDLING
    ===================================== */

    catch (error) {


        console.error(
            "Fact checking failed:",
            error
        );


        let errorMessage =
            "Something went wrong. Please try again.";


        /* Timeout */

        if (
            error.name === "AbortError"
        ) {

            errorMessage =
                "The request timed out. The AI server may be waking up. Please try again.";

        }


        /* Server not found */

        else if (
            error.message.includes("404")
        ) {

            errorMessage =
                "Server not found. Please check whether your Hugging Face Space is running.";

        }


        /* Internal server error */

        else if (
            error.message.includes("500")
        ) {

            errorMessage =
                "The AI model encountered an internal server error. Please try again.";

        }


        /* Network error */

        else if (
            error.message.includes("Failed to fetch")
        ) {

            errorMessage =
                "Unable to connect to the AI server. Please check your internet connection and try again.";

        }


        /* Show error */

        setStatus(
            errorMessage,
            "error",
            false
        );


    }


    /* =====================================
       FINAL UI RESET
    ===================================== */

    finally {


        /* Enable button again */

        updateButtonState();


        /* Restore button */

        checkButton.innerHTML =
            "✨ Analyze Health Claim";


    }


}

/* =========================================
HUGGING FACE API
========================================= */

async function callCustomModelAPI(
    claim,
    signal
) {


    const response =
        await fetch(
            HUGGINGFACE_ENDPOINT,
            {

                method: "POST",


                headers: {

                    "Content-Type":
                        "application/json"

                },


                body:

                    JSON.stringify({

                        claim: claim

                    }),


                signal: signal

            }
        );


    /* Check if request was aborted */

    if (signal.aborted) {

        throw new DOMException(
            "Request aborted",
            "AbortError"
        );

    }


    /* Check server response */

    if (!response.ok) {

        throw new Error(
            `API Request failed with status: ${response.status}`
        );

    }


    /* Convert response to JSON */

    const result =
        await response.json();


    /* Check API error */

    if (result.error) {

        throw new Error(
            result.error
        );

    }


    /* Return AI verdict */

    return result.verdict;


}

/* =========================================
STATUS MESSAGE
========================================= */

let statusTimeout;

/* Show status message */

function setStatus(
    message,
    type,
    showSpinner
) {


    /* Remove previous timeout */

    clearTimeout(statusTimeout);


    /* Remove old status classes */

    statusMessage.classList.remove(
        "loading",
        "success",
        "error",
        "hidden"
    );


    /* Add new status type */

    statusMessage.classList.add(type);


    /* Update message */

    statusText.textContent =
        message;


    /* Spinner visibility */

    if (showSpinner) {

        spinner.style.display =
            "block";

    }

    else {

        spinner.style.display =
            "none";

    }


    /* Hide successful messages */

    if (type === "success") {

        statusTimeout =
            setTimeout(
                () => {

                    statusMessage.classList.add(
                        "hidden"
                    );

                },
                5000
            );

    }


}

/* =========================================
FORMAT AI RESPONSE
========================================= */

function formatResponse(text) {


    /* Convert response safely to string */

    if (!text) {

        return `
        <p class="response-paragraph">
            The AI model did not return a response.
        </p>
    `;

    }


    /* Split response into lines */

    const lines =
        text
            .split("\n")
            .filter(
                line =>
                    line.trim() !== ""
            );


    let html = "";


    lines.forEach(line => {


        const cleanLine =
            line
                .replace(/\*\*/g, "")
                .trim();


        /* =================================
           VERDICT
        ================================= */

        if (
            cleanLine
                .toLowerCase()
                .startsWith("verdict")
        ) {


            html += `

            <div class="verdict-title">

                ${cleanLine}

            </div>

        `;

        }


        /* =================================
           SECTION HEADINGS
        ================================= */

        else if (
            line.startsWith("**")
        ) {


            html += `

            <div class="section-title">

                ${cleanLine}

            </div>

        `;

        }


        /* =================================
           NORMAL TEXT
        ================================= */

        else {


            html += `

            <p class="response-paragraph">

                ${cleanLine}

            </p>

        `;

        }


    });


    return html;


}

/* =========================================
INITIAL SETUP
========================================= */

updateButtonState();

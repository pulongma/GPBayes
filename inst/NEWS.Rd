\name{NEWS}
\title{News for Package \pkg{GPBayes}}
\newcommand{\ghpr}{\href{https://github.com/pulongma/GPBayes/pull/#1}{##1}}
\newcommand{\ghit}{\href{https://github.com/pulongma/GPBayes/issues/#1}{##1}}

\section{Changes in GPBayes version 0.1.0-6 (2024-4-24)}{
	\itemize{
		\item Added checks for the arguments in \link{gp.mcmc} when a ARD type of covariances is specified.
		\item Added the function \link{gp.condsim} to perform conditional simulation. 
	}
}

\section{Changes in GPBayes version 0.1.0-5.1 (2023-1-31)}{
	\itemize{
		\item Updated C++ specification to C++17 to please CRAN.  
	}
}

\section{Changes in GPBayes version 0.1.0-5 (2023-1-12)}{
	\itemize{
		\item Fixed the issue in the configure file.  
	}
}

\section{Changes in GPBayes version 0.1.0-4 (2022-8-01)}{
	\itemize{
		\item Updated documentation for several functions.  
		\item Added limiting values for the Matérn and Confluent Hypergeometric covariance functions when their range parameters go to zero.
	}
}

\section{Changes in GPBayes version 0.1.0-3 (2021-12-01)}{
	\itemize{
		\item Added the package NEWS. 
		\item Added messages when loading the package in R.
	}
}

\section{Changes in GPBayes version 0.1.0-2.2 (2021-10-15)}{
	\itemize{
		\item License has been change to GPL 2 and GPL-3.
	}
}

\section{Changes in GPBayes version 0.1.0-2.1 (2021-10-8)}{
	\itemize{
		\item Citation info is corrected.  
	}
}

\section{Changes in GPBayes version 0.1.0-2 (2021-09-13)}{
	\itemize{
		\item Initial submission to CRAN.
		\item Release of the pacakage on Github publicly.
		\item Full documentation and examples have been added to core functions.
	}
}

\section{Changes in GPBayes version 0.1.0-1 (2020-04-02)}{
	\itemize{
		\item Initial release of the package to Github as a private package.
	}
}

type ErrorMessageProps = {
  error: string | null;
};

const ErrorMessage = ({ error }: ErrorMessageProps): JSX.Element => {
  if (!error) return <></>;

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 md:top-8 max-w-md p-4 rounded-md border border-red-300 bg-red-50 flex items-center">
      <svg className="h-5 w-5 text-red-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
      <div className="ml-3">
        <h3 className="text-sm font-medium text-red-800">Error</h3>
        <div className="mt-1 text-sm text-red-700">{error}</div>
      </div>
    </div>
  );
};

export default ErrorMessage;

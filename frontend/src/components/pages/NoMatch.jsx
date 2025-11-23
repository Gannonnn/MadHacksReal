import { Link } from "react-router";
import { Button } from 'react-bootstrap';

function NoMatch() {
    return (
        <div className="text-center">
            <h2>Page was not found</h2>
            <Button as={Link} className="my-5" to="/" variant="outline-secondary">Back to home</Button>
        </div>
    );
}

export default NoMatch;

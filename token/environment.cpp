
#include "environment.h"

void Environment::randomizeToken(){
    while(true){
        token = Pos(randomN(size), randomN(size));
        if(token != agent) break;
    }
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    agent = Pos(size/2, size/2);
    randomizeToken();
}

string Environment::toString(){
    return "Time: " + to_string(timeIndex) + " Agent: " + to_string(agent.x) + " " + to_string(agent.y) +
                                           + " Token: " + to_string(token.x) + " " + to_string(token.y);
}

string Environment::toCode(){
    return to_string(timeIndex) + " " + to_string(agent.x) + " " + to_string(agent.y) + " " + to_string(token.x) + " " + to_string(token.y) + "\n";
}

vector<int> Environment::validActions(){
    return vector<int>{0, 1, 2, 3};
}

double Environment::makeAction(int action){
    double reward = 0;

    agent = agent.shift(action);
    if(!agent.inBounds()){
        reward = -1;
        endState = true;
    }
    else if(agent == token){
        reward = 1;
        randomizeToken();

        // place token close to agent if captured from the right
        // reward = 1;
        // int close = 1;
        // if(action == 2 && randUniform() < 0.8){
        //     while(true){
        //         token = Pos(agent.x + randomN(2*close+1)-close, agent.y + randomN(2*close+1)-close);
        //         if(token.inBounds() && token != agent) break;
        //     }
        // }
        // else{
        //     randomizeToken();
        // }
    }

    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }

    // int symID = randomSym();
    // agent = transform(agent, symID);
    // token = transform(token, symID);

    return reward;
}

void Environment::getFeatures(double* features){
    features[0] = agent.x;
    features[1] = agent.y;
    features[2] = token.x;
    features[3] = token.y;
}

int randomSym(){
    return randomN(8);
}

int symAction(int action, int symID){
    return (symDir[symID][0]*action + symDir[symID][1] + 4) % 4;
}

Pos transform(Pos p, int symID){
    int x = sym[symID][0][0]*p.x + sym[symID][0][1]*p.y + sym[symID][0][2];
    int y = sym[symID][1][0]*p.x + sym[symID][1][1]*p.y + sym[symID][1][2];
    return Pos(x, y);
}
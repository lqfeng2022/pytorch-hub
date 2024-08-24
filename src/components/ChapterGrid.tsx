import { Box, Divider, Flex, Heading, Text } from '@chakra-ui/react';
import ChapterCover from './ChapterCover';
import ChapterList from './ChapterList';
import ChapterQuote from './ChapterQuote';
import AIDefinition from './AIDefinition';
import AIWhy from './AIWhy';
import AIHistroy from './AIHistroy';
import MLDefinition from './MLDefinition';
import MLComparing from './MLComparing';
import MLModels from './MLModels';
import DLDefinition from './DLDefinition';
import DLRelationship from './DLRelationship';
import DLComparing from './DLComparing';
import DLApps from './DLApps';
import NNDefinition from './NNDefinition';
import NNArchitecture from './NNArchitecture';
import Neurons from './Neurons';
import Libraries from './Libraries';
import PTDefinition from './PTDefinition';
import PTTrends from './PTTrends';
import PTCompanies from './PTCompanies';
import Prerequisites from './Prerequisites';
import TakeCourse from './TakeCourse';
import PTFeatures from './PTFeatures';

const ChapterGrid = () => {
  const title = {name: "Chapter 0: Artificial Intelligence"}

  return (
    <Box p={3} maxW='900px'>
      <Heading size='xl' textAlign='center' py={3}>{title.name}</Heading>
      <ChapterQuote/>
      <ChapterCover/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList/>
      {/* 1. Artificial Intelligence */}
      <Text as='b' fontSize='xl'>1. Artificial Intelligence</Text>
      <AIDefinition/>
      <AIWhy/>
      <AIHistroy/>
      {/* 2. Machine Learning */}
      <Text as='b' fontSize='xl'>2. Machine Learning</Text>
      <MLDefinition/>
      <MLComparing/>
      <MLModels/>
      {/* 3. Deep Learning */}
      <Text as='b' fontSize='xl'>3. Deep Learning</Text>
      <DLDefinition/>
      <DLRelationship/>
      <DLComparing/>
      <DLApps/>
      {/* 4. Neural Network */}
      <Text as='b' fontSize='xl'>4. Neural Network</Text>
      <NNDefinition/>
      <NNArchitecture/>
      <Neurons/>
      {/* 5. Frameworks and Libraries */}
      <Text as='b' fontSize='xl'>5. FRAMEWORKs and LIBRARIES</Text>
      <Libraries/>
      {/* 6. PyTorch */}
      <Text as='b' fontSize='xl'>6. PyTorch</Text>
      <PTDefinition/>
      <PTFeatures/>
      <PTTrends/>
      <PTCompanies/>
      {/* 7. PREREQUISITEs */}
      <Prerequisites/>
      {/* 8. How to take this course */}
      <TakeCourse/>
    </Box>
  )
}

export default ChapterGrid